//
//! Copyright 2021 Alibaba Group Holding Limited.
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

use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use dyn_type::{object, Object};
use graph_proxy::apis::graph::PKV;
use graph_proxy::apis::partitioner::{PartitionInfo, PartitionedData};
use graph_proxy::apis::{get_graph, ClusterInfo, Edge, QueryParams, Vertex, ID};
use ir_common::error::{ParsePbError, ParsePbResult};
use ir_common::generated::algebra as algebra_pb;
use ir_common::generated::physical as pb;
use ir_common::{KeyId, NameOrId};

use crate::error::{FnGenError, FnGenResult};
use crate::process::record::Record;
use crate::router::Router;

#[derive(Debug)]
pub enum SourceType {
    Vertex,
    Edge,
    Table,
    Dummy,
}

/// Source Operator, fetching a source from the (graph) database
#[derive(Debug)]
pub struct SourceOperator {
    query_params: QueryParams,
    src: Option<HashMap<u64, Vec<ID>>>,
    primary_key_values: Option<Vec<PKV>>,
    alias: Option<KeyId>,
    source_type: SourceType,
    // to specify if the source is a fusion of scan and count
    // currently, it may fuse: 1) scan + count; 2) index_scan + count
    is_count_only: bool,
}

impl Default for SourceOperator {
    fn default() -> Self {
        SourceOperator {
            query_params: QueryParams::default(),
            src: None,
            primary_key_values: None,
            alias: None,
            source_type: SourceType::Dummy,
            is_count_only: false,
        }
    }
}

impl SourceOperator {
    pub fn new<P: PartitionInfo, C: ClusterInfo>(
        op: pb::PhysicalOpr, partitioner: Arc<dyn Router<P = P, C = C>>,
    ) -> FnGenResult<Self> {
        let op_kind = op.try_into()?;
        match op_kind {
            pb::physical_opr::operator::OpKind::Scan(mut scan) => {
                if let Some(index_predicate) = scan.idx_predicate.take() {
                    let ip = index_predicate.clone();
                    let ip2 = index_predicate.clone();
                    let mut source_op = SourceOperator::try_from(scan)?;
                    let global_ids: Vec<ID> = <Vec<i64>>::try_from(ip)?
                        .into_iter()
                        .map(|i| i as ID)
                        .collect();
                    if !global_ids.is_empty() {
                        // query by global_ids
                        source_op.set_src(global_ids, partitioner)?;
                        debug!("Runtime source op of indexed scan of global ids {:?}", source_op);
                    } else {
                        // query by indexed_scan
                        let primary_key_values = <Vec<Vec<(NameOrId, Object)>>>::try_from(ip2)?;
                        let pkvs = primary_key_values
                            .into_iter()
                            .map(|pkv| PKV::from(pkv))
                            .collect();
                        source_op.primary_key_values = Some(pkvs);
                        debug!("Runtime source op of indexed scan {:?}", source_op);
                    }
                    Ok(source_op)
                } else {
                    let source_op = SourceOperator::try_from(scan)?;
                    debug!("Runtime source op of scan {:?}", source_op);
                    Ok(source_op)
                }
            }
            pb::physical_opr::operator::OpKind::Root(_) => Ok(SourceOperator::default()),
            _ => Err(ParsePbError::from("algebra_pb op is not a source"))?,
        }
    }

    /// Assign source vertex ids for each worker to call get_vertex
    fn set_src<P: PartitionInfo, C: ClusterInfo>(
        &mut self, ids: Vec<ID>, partitioner: Arc<dyn Router<P = P, C = C>>,
    ) -> ParsePbResult<()> {
        let mut partitions = HashMap::new();
        for id in ids {
            match partitioner.route(id.get_partition_key_id()) {
                Ok(wid) => {
                    partitions
                        .entry(wid)
                        .or_insert_with(Vec::new)
                        .push(id);
                }
                Err(err) => Err(ParsePbError::Unsupported(format!(
                    "get server id failed in graph_partition_manager in source op {:?}",
                    err
                )))?,
            }
        }
        self.src = Some(partitions);
        Ok(())
    }
}

impl SourceOperator {
    pub fn gen_source(self, worker_index: usize) -> FnGenResult<Box<dyn Iterator<Item = Record> + Send>> {
        let graph = get_graph().ok_or_else(|| FnGenError::NullGraphError)?;

        match self.source_type {
            SourceType::Vertex => {
                let mut v_source = Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Vertex> + Send>;
                if let Some(seeds) = &self.src {
                    if let Some(src) = seeds.get(&(worker_index as u64)) {
                        if !src.is_empty() {
                            v_source = graph.get_vertex(src, &self.query_params)?;
                        }
                    }
                    if self.is_count_only {
                        let count = v_source.count() as u64;
                        return Ok(Box::new(
                            vec![Record::new(object!(count), self.alias.clone())].into_iter(),
                        ));
                    }
                } else if let Some(pkvs) = &self.primary_key_values {
                    if self.query_params.labels.is_empty() {
                        Err(FnGenError::unsupported_error(
                            "Empty label in `IndexScan` self.query_params.labels",
                        ))?
                    }
                    let mut source_vertices = vec![];
                    for label in &self.query_params.labels {
                        for pkv in pkvs {
                            if let Some(v) = graph.index_scan_vertex(*label, pkv, &self.query_params)? {
                                source_vertices.push(v);
                            }
                        }
                    }
                    if self.is_count_only {
                        let count = source_vertices.len() as u64;
                        return Ok(Box::new(
                            vec![Record::new(object!(count), self.alias.clone())].into_iter(),
                        ));
                    }
                    v_source = Box::new(source_vertices.into_iter());
                } else {
                    if self.is_count_only {
                        let count = graph.count_vertex(&self.query_params)?;
                        return Ok(Box::new(
                            vec![Record::new(object!(count), self.alias.clone())].into_iter(),
                        ));
                    } else {
                        v_source = graph.scan_vertex(&self.query_params)?;
                    }
                };
                Ok(Box::new(v_source.map(move |v| Record::new(v, self.alias.clone()))))
            }
            SourceType::Edge => {
                let mut e_source = Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Edge> + Send>;
                if let Some(ref seeds) = self.src {
                    if let Some(src) = seeds.get(&(worker_index as u64)) {
                        if !src.is_empty() {
                            e_source = graph.get_edge(src, &self.query_params)?;
                        }
                    }
                    if self.is_count_only {
                        let count = e_source.count() as u64;
                        return Ok(Box::new(
                            vec![Record::new(object!(count), self.alias.clone())].into_iter(),
                        ));
                    }
                } else {
                    if self.is_count_only {
                        let count = graph.count_edge(&self.query_params)?;
                        return Ok(Box::new(
                            vec![Record::new(object!(count), self.alias.clone())].into_iter(),
                        ));
                    } else {
                        e_source = graph.scan_edge(&self.query_params)?;
                    }
                }
                Ok(Box::new(e_source.map(move |e| Record::new(e, self.alias.clone()))))
            }

            SourceType::Table => Err(FnGenError::unsupported_error(
                "neither `Edge` nor `Vertex` but `Table` type `Source` opr",
            ))?,
            SourceType::Dummy => {
                // a dummy record to trigger the computation
                Ok(Box::new(vec![Record::new(Object::None, None)].into_iter())
                    as Box<dyn Iterator<Item = Record> + Send>)
            }
        }
    }
}

impl TryFrom<pb::Scan> for SourceOperator {
    type Error = ParsePbError;

    fn try_from(scan_pb: pb::Scan) -> Result<Self, Self::Error> {
        let scan_opt: algebra_pb::scan::ScanOpt = unsafe { ::std::mem::transmute(scan_pb.scan_opt) };
        let source_type = match scan_opt {
            algebra_pb::scan::ScanOpt::Vertex => SourceType::Vertex,
            algebra_pb::scan::ScanOpt::Edge => SourceType::Edge,
            algebra_pb::scan::ScanOpt::Table => SourceType::Table,
        };
        let query_params = QueryParams::try_from(scan_pb.params)?;
        Ok(SourceOperator {
            query_params,
            src: None,
            primary_key_values: None,
            alias: scan_pb.alias,
            source_type,
            is_count_only: scan_pb.is_count_only,
        })
    }
}
