//
//! Copyright 2022 Alibaba Group Holding Limited.
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//! http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

use std::convert::TryInto;
use std::sync::Arc;

use graph_proxy::apis::cluster_info::ClusterInfo;
use graph_proxy::apis::partitioner::PartitionInfo;
use ir_common::error::ParsePbError;
use ir_common::generated::algebra as algebra_pb;
use ir_common::generated::algebra::join::JoinKind;
use ir_common::generated::physical as pb;
use ir_common::generated::physical::physical_opr::operator::OpKind;
use pegasus::api::function::*;
use pegasus::api::{
    Collect, CorrelatedSubTask, Count, Dedup, Filter, Fold, FoldByKey, HasAny, IterCondition, Iteration,
    Join, KeyBy, Limit, Map, Merge, Sink, SortBy, SortLimitBy,
};
use pegasus::stream::Stream;
use pegasus::{BuildJobError, Worker};
use pegasus_server::job::{JobAssembly, JobDesc};
use pegasus_server::job_pb as server_pb;
use prost::Message;

use crate::error::{FnExecError, FnGenError, FnGenResult};
use crate::process::functions::{ApplyGen, CompareFunction, FoldGen, GroupGen, JoinKeyGen, KeyFunction};
use crate::process::operator::accum::accumulator::Accumulator;
use crate::process::operator::accum::{SampleAccum, SampleAccumFactoryGen};
use crate::process::operator::filter::FilterFuncGen;
use crate::process::operator::flatmap::FlatMapFuncGen;
use crate::process::operator::keyed::KeyFunctionGen;
use crate::process::operator::map::{FilterMapFuncGen, MapFuncGen};
use crate::process::operator::shuffle::RecordRouter;
use crate::process::operator::sink::{SinkGen, Sinker};
use crate::process::operator::sort::CompareFunctionGen;
use crate::process::operator::source::SourceOperator;
use crate::process::record::{Record, RecordKey};
use crate::router::{DefaultRouter, Router};

type RecordMap = Box<dyn MapFunction<Record, Record>>;
type RecordFilterMap = Box<dyn FilterMapFunction<Record, Record>>;
type RecordFlatMap = Box<dyn FlatMapFunction<Record, Record, Target = DynIter<Record>>>;
type RecordFilter = Box<dyn FilterFunction<Record>>;
type RecordLeftJoin = Box<dyn ApplyGen<Record, Vec<Record>, Option<Record>>>;
type RecordShuffle = Box<dyn RouteFunction<Record>>;
type RecordCompare = Box<dyn CompareFunction<Record>>;
type RecordJoin = Box<dyn JoinKeyGen<Record, RecordKey, Record>>;
type RecordKeySelector = Box<dyn KeyFunction<Record, RecordKey, Record>>;
type RecordGroup = Box<dyn GroupGen<Record, RecordKey, Record>>;
type RecordFold = Box<dyn FoldGen<u64, Record>>;

pub struct IRJobAssembly<P: PartitionInfo, C: ClusterInfo> {
    udf_gen: FnGenerator<P, C>,
}

struct FnGenerator<P: PartitionInfo, C: ClusterInfo> {
    router: Arc<dyn Router<P = P, C = C>>,
}

impl<P: PartitionInfo, C: ClusterInfo> Clone for FnGenerator<P, C> {
    fn clone(&self) -> Self {
        Self { router: self.router.clone() }
    }
}

/// An UDF generator for physical operators,
/// which generates the udf that can be executed by the engine.
impl<P: PartitionInfo, C: ClusterInfo> FnGenerator<P, C> {
    fn new(router: Arc<dyn Router<P = P, C = C>>) -> Self {
        FnGenerator { router }
    }

    fn with(partition_info: Arc<P>, cluster_info: Arc<C>) -> Self {
        let router = Arc::new(DefaultRouter::new(partition_info, cluster_info));
        FnGenerator { router }
    }

    fn gen_source(&self, opr: pb::PhysicalOpr) -> FnGenResult<DynIter<Record>> {
        let worker_id = pegasus::get_current_worker();
        let source_opr = SourceOperator::new(opr, self.router.clone())?;
        Ok(source_opr.gen_source(worker_id.index as usize)?)
    }

    fn gen_shuffle(&self, res: &pb::repartition::Shuffle) -> FnGenResult<RecordShuffle> {
        let p = self.router.clone();
        let record_router = RecordRouter::new(p, res.shuffle_key)?;
        Ok(Box::new(record_router))
    }

    fn gen_project(&self, opr: pb::Project) -> FnGenResult<RecordFilterMap> {
        Ok(opr.gen_filter_map()?)
    }

    fn gen_unfold(&self, opr: pb::Unfold) -> FnGenResult<RecordFlatMap> {
        Ok(opr.gen_flat_map()?)
    }

    fn gen_filter(&self, opr: algebra_pb::Select) -> FnGenResult<RecordFilter> {
        Ok(opr.gen_filter()?)
    }

    fn gen_cmp(&self, opr: algebra_pb::OrderBy) -> FnGenResult<RecordCompare> {
        Ok(opr.gen_cmp()?)
    }

    fn gen_group(&self, opr: pb::GroupBy) -> FnGenResult<RecordGroup> {
        Ok(Box::new(opr))
    }

    fn gen_fold(&self, opr: pb::GroupBy) -> FnGenResult<RecordFold> {
        Ok(Box::new(opr))
    }

    fn gen_apply(&self, opr: pb::Apply) -> FnGenResult<RecordLeftJoin> {
        Ok(Box::new(opr))
    }

    fn gen_join(&self, opr: pb::Join) -> FnGenResult<RecordJoin> {
        Ok(Box::new(opr))
    }

    fn gen_dedup(&self, opr: algebra_pb::Dedup) -> FnGenResult<RecordKeySelector> {
        Ok(opr.gen_key()?)
    }

    fn gen_vertex(&self, opr: pb::GetV) -> FnGenResult<RecordFilterMap> {
        Ok(opr.gen_filter_map()?)
    }

    fn gen_both_vertex(&self, opr: pb::GetV) -> FnGenResult<RecordFlatMap> {
        Ok(opr.gen_flat_map()?)
    }

    fn gen_edge_expand(&self, opr: pb::EdgeExpand) -> FnGenResult<RecordFlatMap> {
        Ok(opr.gen_flat_map()?)
    }

    fn gen_edge_expand_collection(&self, opr: pb::EdgeExpand) -> FnGenResult<RecordFilterMap> {
        Ok(opr.gen_filter_map()?)
    }

    fn gen_path_start(&self, opr: pb::PathExpand) -> FnGenResult<RecordFilterMap> {
        Ok(opr.gen_filter_map()?)
    }

    fn gen_path_end(&self, opr: pb::PathExpand) -> FnGenResult<RecordMap> {
        Ok(opr.gen_map()?)
    }

    fn gen_path_condition(&self, opr: pb::PathExpand) -> FnGenResult<RecordFilter> {
        Ok(opr.gen_filter()?)
    }

    fn gen_coin(&self, opr: algebra_pb::Sample) -> FnGenResult<RecordFilter> {
        Ok(opr.gen_filter()?)
    }

    fn gen_sample(&self, opr: algebra_pb::Sample) -> FnGenResult<SampleAccum> {
        Ok(opr.gen_accum()?)
    }

    fn gen_sink(&self, opr: pb::PhysicalOpr) -> FnGenResult<Sinker> {
        Ok(opr.gen_sink()?)
    }
}

impl<P: PartitionInfo, C: ClusterInfo> IRJobAssembly<P, C> {
    pub fn new(router: Arc<dyn Router<P = P, C = C>>) -> Self {
        let udf_gen = FnGenerator::new(router);
        IRJobAssembly { udf_gen }
    }

    pub fn with(partition_info: Arc<P>, cluster_info: Arc<C>) -> Self {
        let udf_gen = FnGenerator::with(partition_info, cluster_info);
        IRJobAssembly { udf_gen }
    }

    fn install(
        &self, mut stream: Stream<Record>, plan: &[pb::PhysicalOpr],
    ) -> Result<Stream<Record>, BuildJobError> {
        let mut prev_op_kind = pb::physical_opr::operator::OpKind::Root(pb::Root {});
        for op in &plan[..] {
            let op_kind = op.try_into().map_err(|e| FnGenError::from(e))?;
            match op_kind {
                OpKind::Repartition(repartition) => {
                    let repartition_strategy = repartition.strategy.as_ref().ok_or_else(|| {
                        FnGenError::from(ParsePbError::EmptyFieldError(
                            "Empty repartition strategy".to_string(),
                        ))
                    })?;
                    match repartition_strategy {
                        pb::repartition::Strategy::ToAnother(shuffle) => {
                            let router = self.udf_gen.gen_shuffle(shuffle)?;
                            stream = stream.repartition(move |t| router.route(t));
                        }
                        pb::repartition::Strategy::ToOthers(_) => stream = stream.broadcast(),
                    }
                }
                OpKind::Project(project) => {
                    let func = self.udf_gen.gen_project(project)?;
                    stream = stream.filter_map_with_name("Project", move |input| func.exec(input))?;
                }
                OpKind::Select(select) => {
                    let func = self.udf_gen.gen_filter(select)?;
                    stream = stream.filter(move |input| func.test(input))?;
                }
                OpKind::Unfold(unfold) => {
                    let func = self.udf_gen.gen_unfold(unfold)?;
                    stream = stream.flat_map_with_name("Unfold", move |input| func.exec(input))?;
                }
                OpKind::Limit(limit) => {
                    let range = limit.range.ok_or_else(|| {
                        FnGenError::from(ParsePbError::EmptyFieldError("pb::Limit::range".to_string()))
                    })?;
                    // e.g., `limit(10)` would be translate as `Range{lower=0, upper=10}`
                    if range.upper <= range.lower || range.lower != 0 {
                        Err(FnGenError::from(ParsePbError::ParseError(format!(
                            "range {:?} in Limit Operator",
                            range
                        ))))?;
                    }
                    stream = stream.limit(range.upper as u32)?;
                }
                OpKind::OrderBy(order) => {
                    let cmp = self.udf_gen.gen_cmp(order.clone())?;
                    if let Some(range) = order.limit {
                        if range.upper <= range.lower || range.lower != 0 {
                            Err(FnGenError::from(ParsePbError::ParseError(format!(
                                "range {:?} in Order Operator",
                                range
                            ))))?;
                        }
                        stream = stream.sort_limit_by(range.upper as u32, move |a, b| cmp.compare(a, b))?;
                    } else {
                        stream = stream.sort_by(move |a, b| cmp.compare(a, b))?;
                    }
                }
                OpKind::GroupBy(group) => {
                    if group.mappings.is_empty() {
                        // fold case
                        let fold = self.udf_gen.gen_fold(group)?;
                        if let server_pb::AccumKind::Cnt = fold.get_accum_kind() {
                            let fold_map = fold.gen_fold_map()?;
                            stream = stream
                                .count()?
                                .map(move |cnt| fold_map.exec(cnt))?
                                .into_stream()?;
                        } else {
                            // TODO: optimize this by fold_partiton + fold
                            let fold_accum = fold.gen_fold_accum()?;
                            stream = stream
                                .fold(fold_accum, || {
                                    |mut accumulator, next| {
                                        accumulator.accum(next)?;
                                        Ok(accumulator)
                                    }
                                })?
                                .map(move |mut accum| Ok(accum.finalize()?))?
                                .into_stream()?;
                        }
                    } else {
                        // group case
                        let group = self.udf_gen.gen_group(group)?;
                        let group_key = group.gen_group_key()?;
                        let group_accum = group.gen_group_accum()?;
                        let group_map = group.gen_group_map()?;
                        stream = stream
                            .key_by(move |record| group_key.get_kv(record))?
                            .fold_partition_by_key(group_accum, || {
                                |mut accumulator, next| {
                                    accumulator.accum(next)?;
                                    Ok(accumulator)
                                }
                            })?
                            .unfold(|kv_map| {
                                Ok(kv_map
                                    .into_iter()
                                    .map(|(key, mut accumulator)| {
                                        accumulator.finalize().map(|value| (key, value))
                                    })
                                    .collect::<Result<Vec<_>, _>>()?
                                    .into_iter())
                            })?
                            .map(move |key_value| group_map.exec(key_value))?;
                    }
                }
                OpKind::Dedup(dedup) => {
                    let selector = self.udf_gen.gen_dedup(dedup)?;
                    stream = stream
                        .key_by(move |record| selector.get_kv(record))?
                        .dedup()?
                        .map(|pair| Ok(pair.value))?;
                }
                OpKind::Union(union) => {
                    let (mut ori_stream, sub_stream) = stream.copied()?;
                    stream = self.install(sub_stream, &union.sub_plans[0].plan[..])?;
                    for subtask in &union.sub_plans[1..] {
                        let copied = ori_stream.copied()?;
                        ori_stream = copied.0;
                        stream = self
                            .install(copied.1, &subtask.plan[..])?
                            .merge(stream)?;
                    }
                }
                OpKind::Apply(apply) => {
                    if apply.keys.is_empty() {
                        // apply
                        let apply_gen = self.udf_gen.gen_apply(apply.clone())?;
                        let join_kind = apply_gen.get_join_kind();
                        let join_func = apply_gen.gen_left_join_func()?;
                        let sub_task = apply.sub_plan.as_ref().ok_or_else(|| {
                            BuildJobError::Unsupported("Task is missing in Apply".to_string())
                        })?;
                        stream = match join_kind {
                            JoinKind::Semi => stream
                                .apply(|sub_start| {
                                    let has_sub = self
                                        .install(sub_start, &sub_task.plan[..])?
                                        .any()?;
                                    Ok(has_sub)
                                })?
                                .filter_map(
                                    move |(parent, has_sub)| {
                                        if has_sub {
                                            Ok(Some(parent))
                                        } else {
                                            Ok(None)
                                        }
                                    },
                                )?,
                            JoinKind::Anti => stream
                                .apply(|sub_start| {
                                    let has_sub = self
                                        .install(sub_start, &sub_task.plan[..])?
                                        .any()?;
                                    Ok(has_sub)
                                })?
                                .filter_map(
                                    move |(parent, has_sub)| {
                                        if has_sub {
                                            Ok(None)
                                        } else {
                                            Ok(Some(parent))
                                        }
                                    },
                                )?,
                            JoinKind::Inner | JoinKind::LeftOuter => stream
                                .apply(|sub_start| {
                                    let sub_end = self
                                        .install(sub_start, &sub_task.plan[..])?
                                        .collect::<Vec<Record>>()?;
                                    Ok(sub_end)
                                })?
                                .filter_map(move |(parent, sub)| join_func.exec(parent, sub))?,
                            _ => Err(BuildJobError::Unsupported(format!(
                                "Do not support join_kind {:?} in Apply",
                                join_kind
                            )))?,
                        };
                    } else {
                        // segment apply
                        Err(FnGenError::unsupported_error("SegmentApply Operator"))?
                    }
                }
                OpKind::Join(join) => {
                    let joiner = self.udf_gen.gen_join(join.clone())?;
                    let left_key_selector = joiner.gen_left_kv_fn()?;
                    let right_key_selector = joiner.gen_right_kv_fn()?;
                    let join_kind = joiner.get_join_kind();
                    let left_task = join
                        .left_plan
                        .as_ref()
                        .ok_or_else(|| FnGenError::ParseError("left_task is missing in merge".into()))?;
                    let right_task = join
                        .right_plan
                        .as_ref()
                        .ok_or_else(|| FnGenError::ParseError("right_task is missing in merge".into()))?;
                    let (left_stream, right_stream) = stream.copied()?;
                    let left_stream = self
                        .install(left_stream, &left_task.plan[..])?
                        .key_by(move |record| left_key_selector.get_kv(record))?;
                    let right_stream = self
                        .install(right_stream, &right_task.plan[..])?
                        .key_by(move |record| right_key_selector.get_kv(record))?;
                    stream = match join_kind {
                        JoinKind::Inner => left_stream
                            .inner_join(right_stream)?
                            .map(|(left, right)| Ok(left.value.join(right.value, None)))?,
                        JoinKind::LeftOuter => {
                            left_stream
                                .left_outer_join(right_stream)?
                                .map(|(left, right)| {
                                    let left = left.ok_or_else(|| {
                                        FnExecError::unexpected_data_error(
                                            "left is None in left outer join",
                                        )
                                    })?;
                                    if let Some(right) = right {
                                        // TODO(bingqing): Specify HeadJoinOpt if necessary
                                        Ok(left.value.join(right.value, None))
                                    } else {
                                        Ok(left.value)
                                    }
                                })?
                        }
                        JoinKind::RightOuter => left_stream
                            .right_outer_join(right_stream)?
                            .map(|(left, right)| {
                                let right = right.ok_or_else(|| {
                                    FnExecError::unexpected_data_error("right is None in right outer join")
                                })?;
                                if let Some(left) = left {
                                    Ok(left.value.join(right.value, None))
                                } else {
                                    Ok(right.value)
                                }
                            })?,
                        JoinKind::FullOuter => {
                            left_stream
                                .full_outer_join(right_stream)?
                                .map(|(left, right)| match (left, right) {
                                    (Some(left), Some(right)) => Ok(left.value.join(right.value, None)),
                                    (Some(left), None) => Ok(left.value),
                                    (None, Some(right)) => Ok(right.value),
                                    (None, None) => {
                                        unreachable!()
                                    }
                                })?
                        }
                        JoinKind::Semi => left_stream
                            .semi_join(right_stream)?
                            .map(|left| Ok(left.value))?,
                        JoinKind::Anti => left_stream
                            .anti_join(right_stream)?
                            .map(|left| Ok(left.value))?,
                        JoinKind::Times => Err(BuildJobError::Unsupported(
                            "JoinKind of Times is not supported yet".to_string(),
                        ))?,
                    }
                }
                OpKind::Intersect(intersect) => {
                    // The intersect op can be:
                    //     1) EdgeExpand with Opt = ExpandV, which is to expand and intersect on id-only vertices;
                    //     2) EdgeExpand with Opt = ExpandE, which is to expand and intersect on edges (not supported yet);
                    // Specifically,
                    //     1) if we want to expand and intersect on vertices, while there are some further filters on the intersected vertices,
                    //        this would be translated into Intersect(EdgeExpand(V), EdgeExpand(V)) + Unfold + Select in physical plan for now.
                    //     2) on distributed graph database, the intersect op exists together with the `Repartition` op in subplans.
                    let mut intersected_expands = vec![];
                    for mut subplan in intersect.sub_plans {
                        if subplan.plan.len() > 2 {
                            Err(FnGenError::unsupported_error(&format!(
                                "subplan in pb::Intersect::plan {:?}",
                                subplan,
                            )))?
                        }
                        let last_op = subplan.plan.pop().ok_or_else(|| {
                            FnGenError::from(ParsePbError::EmptyFieldError(
                                "subplan in pb::Intersect::plan".to_string(),
                            ))
                        })?;
                        let last_op_kind = last_op
                            .try_into()
                            .map_err(|e| FnGenError::from(e))?;
                        match last_op_kind {
                            OpKind::Edge(expand) => {
                                // the case of expand id-only vertex
                                let repartition = if let Some(prev) = subplan.plan.last() {
                                    if let OpKind::Repartition(edge_expand_repartition) = prev
                                        .try_into()
                                        .map_err(|e| FnGenError::from(e))?
                                    {
                                        subplan.plan.pop();
                                        Some(edge_expand_repartition)
                                    } else {
                                        Err(FnGenError::unsupported_error(&format!(
                                            "subplan in pb::Intersect::plan {:?}",
                                            subplan,
                                        )))?
                                    }
                                } else {
                                    None
                                };
                                intersected_expands.push((repartition, expand));
                            }
                            _ => Err(FnGenError::unsupported_error(&format!(
                                "Opr in Intersection to intersect: {:?}",
                                last_op_kind
                            )))?,
                        }
                    }
                    // intersect of edge_expands
                    for (repartition, expand_intersect_opr) in intersected_expands {
                        if let Some(repartition) = repartition {
                            stream = self.install(stream, &vec![repartition.into()])?;
                        }
                        let expand_func = self
                            .udf_gen
                            .gen_edge_expand_collection(expand_intersect_opr)?;
                        stream = stream.filter_map_with_name("ExpandIntersect", move |input| {
                            expand_func.exec(input)
                        })?;
                    }
                }
                OpKind::Vertex(vertex) => {
                    let vertex_opt: algebra_pb::get_v::VOpt = unsafe { std::mem::transmute(vertex.opt) };
                    match vertex_opt {
                        algebra_pb::get_v::VOpt::Both => {
                            let func = self.udf_gen.gen_both_vertex(vertex)?;
                            stream = stream.flat_map_with_name("GetV", move |input| func.exec(input))?;
                        }
                        _ => {
                            let func = self.udf_gen.gen_vertex(vertex)?;
                            stream = stream.filter_map_with_name("GetV", move |input| func.exec(input))?;
                        }
                    }
                }
                OpKind::Edge(edge) => {
                    let func = self.udf_gen.gen_edge_expand(edge)?;
                    stream = stream.flat_map_with_name("EdgeExpand", move |input| func.exec(input))?;
                }
                OpKind::Path(path) => {
                    let mut base = path.base.clone().ok_or_else(|| {
                        FnGenError::from(ParsePbError::EmptyFieldError("pb::PathExpand::base".to_string()))
                    })?;
                    let range = path.hop_range.as_ref().ok_or_else(|| {
                        FnGenError::from(ParsePbError::EmptyFieldError(
                            "pb::PathExpand::hop_range".to_string(),
                        ))
                    })?;
                    if range.upper <= range.lower || range.lower < 0 || range.upper <= 0 {
                        Err(FnGenError::from(ParsePbError::ParseError(format!(
                            "range {:?} in PathExpand Operator",
                            range
                        ))))?;
                    }
                    // path start
                    let path_start_func = self.udf_gen.gen_path_start(path.clone())?;
                    stream = stream
                        .filter_map_with_name("PathStart", move |input| path_start_func.exec(input))?;
                    // path base expand
                    let mut base_expand_plan = vec![];
                    // process edge_expand, with opt = ExpandV given by physical plan.
                    if let Some(edge_expand) = base.edge_expand.take() {
                        if pb::path_expand::ResultOpt::AllVE
                            == unsafe { std::mem::transmute(path.result_opt) }
                        {
                            // the case when base expand needs to expand edges + vertices
                            let mut edge_expand_e = edge_expand.clone();
                            edge_expand_e.expand_opt = pb::edge_expand::ExpandOpt::Edge as i32;
                            let alias = edge_expand_e.alias.take();
                            let get_v = pb::GetV {
                                opt: pb::get_v::VOpt::Other as i32,
                                tag: None,
                                params: None,
                                alias,
                            };
                            base_expand_plan.push(edge_expand_e.into());
                            base_expand_plan.push(get_v.into());
                        } else {
                            // the case when base expand needs to expand vertices
                            base_expand_plan.push(edge_expand.into());
                        }
                    } else {
                        Err(FnGenError::from(ParsePbError::ParseError(format!(
                            "empty EdgeExpand of ExpandBase in PathExpand Operator {:?}",
                            base
                        ))))?;
                    }
                    if let OpKind::Repartition(_) = &prev_op_kind {
                        // the case when base expand needs repartition
                        base_expand_plan.push(
                            pb::Repartition {
                                strategy: Some(pb::repartition::Strategy::ToAnother(
                                    pb::repartition::Shuffle { shuffle_key: None },
                                )),
                            }
                            .into(),
                        );
                    }
                    // process get_v, with opt = Self, given by physical plan (to deal with filtering on vertices).
                    if let Some(getv) = base.get_v.take() {
                        base_expand_plan.push(getv.clone().into());
                    }

                    for _ in 0..range.lower {
                        stream = self.install(stream, &base_expand_plan)?;
                    }
                    let times = range.upper - range.lower - 1;
                    if times > 0 {
                        if path.condition.is_some() {
                            let mut until = IterCondition::max_iters(times as u32);
                            let func = self.udf_gen.gen_path_condition(path.clone())?;
                            until.set_until(func);
                            // Notice that if UNTIL condition set, we expand path without `Emit`
                            stream = stream
                                .iterate_until(until, |start| self.install(start, &base_expand_plan[..]))?;
                        } else {
                            let (mut hop_stream, copied_stream) = stream.copied()?;
                            stream = copied_stream;
                            for _ in 0..times {
                                hop_stream = self.install(hop_stream, &base_expand_plan[..])?;
                                let copied = hop_stream.copied()?;
                                hop_stream = copied.0;
                                stream = stream.merge(copied.1)?;
                            }
                        }
                    }
                    // path end to add path_alias if exists
                    if path.alias.is_some() {
                        let path_end_func = self.udf_gen.gen_path_end(path)?;
                        stream = stream.map_with_name("PathEnd", move |input| path_end_func.exec(input))?;
                    }
                }
                OpKind::Scan(scan) => {
                    let udf_gen = self.udf_gen.clone();
                    stream = stream.flat_map(move |_| {
                        let scan_iter = udf_gen.gen_source(scan.clone().into()).unwrap();
                        Ok(scan_iter)
                    })?;
                }
                OpKind::Sample(sample) => {
                    if let Some(sample_weight) = &sample.sample_weight {
                        if sample_weight.tag.is_some() || sample_weight.property.is_some() {
                            return Err(FnGenError::from(ParsePbError::ParseError(
                                "sample_weight is not supported yet".to_string(),
                            )))?;
                        }
                    }
                    if let Some(sample_type) = &sample.sample_type {
                        match &sample_type.inner {
                            // the case of Coin
                            Some(algebra_pb::sample::sample_type::Inner::SampleByRatio(_)) => {
                                let func = self.udf_gen.gen_coin(sample)?;
                                stream = stream.filter(move |input| func.test(input))?;
                            }
                            // the case of Sample
                            Some(algebra_pb::sample::sample_type::Inner::SampleByNum(_)) => {
                                let partial_sample_accum = self.udf_gen.gen_sample(sample)?;
                                let sample_accum = partial_sample_accum.clone();
                                stream = stream
                                    .fold_partition(partial_sample_accum, move || {
                                        move |mut sample_accum, next| {
                                            sample_accum.accum(next)?;
                                            Ok(sample_accum)
                                        }
                                    })?
                                    .unfold(move |mut sample_accum| Ok(sample_accum.finalize()?))?
                                    .fold(sample_accum, move || {
                                        move |mut sample_accum, next| {
                                            sample_accum.accum(next)?;
                                            Ok(sample_accum)
                                        }
                                    })?
                                    .unfold(move |mut sample_accum| Ok(sample_accum.finalize()?))?
                            }
                            None => Err(FnGenError::from(ParsePbError::EmptyFieldError(
                                "pb::Sample::sample_type.inner".to_string(),
                            )))?,
                        }
                    } else {
                        Err(FnGenError::from(ParsePbError::EmptyFieldError(
                            "pb::Sample::sample_type".to_string(),
                        )))?;
                    }
                }
                OpKind::Root(_) => {
                    // do nothing, as it is a dummy node
                }
                OpKind::Sink(_) => {
                    // this would be processed in assemble, and can not be reached when install.
                    Err(FnGenError::unsupported_error("unreachable sink in install"))?
                }
            }

            prev_op_kind = op.try_into().map_err(|e| FnGenError::from(e))?;
        }
        Ok(stream)
    }
}

impl<P: PartitionInfo, C: ClusterInfo> JobAssembly<Record> for IRJobAssembly<P, C> {
    fn assemble(&self, plan: &JobDesc, worker: &mut Worker<Record, Vec<u8>>) -> Result<(), BuildJobError> {
        worker.dataflow(move |input, output| {
            let physical_plan = decode::<pb::PhysicalPlan>(&plan.plan)?;
            if log_enabled!(log::Level::Debug) && pegasus::get_current_worker().index == 0 {
                debug!("{:#?}", physical_plan);
            }
            // input from a dummy record to trigger the computation
            let source = input.input_from(vec![Record::default()])?;
            let plan_len = physical_plan.plan.len();
            let stream = self.install(source, &physical_plan.plan[0..plan_len - 1])?;
            let sink_opr = physical_plan.plan.last().ok_or_else(|| {
                FnGenError::from(ParsePbError::EmptyFieldError("empty job plan".to_string()))
            })?;
            let ec = self.udf_gen.gen_sink(sink_opr.clone())?;
            match ec {
                Sinker::DefaultSinker(default_sinker) => stream
                    .map(move |record| default_sinker.exec(record))?
                    .sink_into(output),
                #[cfg(feature = "with_v6d")]
                Sinker::GraphSinker(graph_sinker) => {
                    return stream
                        .fold_partition(graph_sinker, || {
                            |mut accumulator, next| {
                                accumulator.accum(next)?;
                                Ok(accumulator)
                            }
                        })?
                        .map(|mut accumulator| Ok(accumulator.finalize()?))?
                        .into_stream()?
                        .map(|_r| Ok(vec![]))?
                        .sink_into(output)
                }
            }
        })
    }
}

#[inline]
fn decode<T: Message + Default>(binary: &[u8]) -> FnGenResult<T> {
    Ok(T::decode(binary)?)
}
