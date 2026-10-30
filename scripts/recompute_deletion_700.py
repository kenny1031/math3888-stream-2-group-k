#!/usr/bin/env python3
"""Recompute exact, unweighted HLJ1-deletion betweenness at score >=700.

Example, from the project root:
    python scripts/recompute_deletion_700.py \
        --edges outputs/jadon/string_v12.0_edges_tau0.7.csv \
        --integrated outputs/integrated/cross_method_evidence.csv

Uses a portable C++17 implementation of Brandes' algorithm for speed.
The original notebook's ratio (after-before)/after is retained under an
unambiguous name; (after-before)/before is also exported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "4932.YMR161W"
THRESHOLD = 700

SEMANTIC_RENAMES = {
    "jadon_degree": "erad_degree",
    "jadon_is_seed": "erad_is_seed",
    "jadon_pagerank_erad": "erad_pagerank_before",
    "jadon_pct_change": "erad_pagerank_pct_change",
    "jadon_bt_before": "erad_sampled_betweenness_before",
    "jadon_bt_pct_change": "erad_sampled_betweenness_pct_change",
    "jadon_d_seed": "erad_seed_distance",
    "jadon_d_hlj1": "erad_hlj1_distance",
    "jadon_J_mean": "erad_jaccard_mean",
    "jadon_J_max": "erad_jaccard_max",
    "jadon_with_HLJ1": "erad_louvain_hlj1_comembership_fraction",
    "jadon_with_ERAD": "erad_louvain_reference_before_fraction",
    "jadon_with_ERAD_after": "erad_louvain_reference_after_fraction",
    "jadon_ERAD_switch": "erad_louvain_reference_fraction_delta",
    "jadon_hlj1_neighbour": "erad_direct_hlj1_neighbour",
    "jadon_essential": "erad_essential",
    "jadon_selected": "erad_screen_selected",
    "jadon_selection_group": "erad_screen_group",
    "jadon_selection_reasons": "erad_screen_reasons",
}
DROP_OLD = {
    "alex_relative_raw", "alex_distance_to_hlj1", "alex_essential",
    "alex_distance_le_2", "alex_available", "jadon_gene", "mcl_gene",
}
METRIC_COLS = [
    "deletion_degree_before", "deletion_degree_after",
    "deletion_distance_to_hlj1",
    "deletion_betweenness_before", "deletion_betweenness_after",
    "deletion_betweenness_delta", "deletion_betweenness_abs_delta",
    "deletion_betweenness_ratio_to_after",  # EXACT original notebook formula
    "deletion_betweenness_ratio_to_before", # standard baseline-relative alternative
    "deletion_betweenness_pct_to_after",
    "deletion_betweenness_pct_to_before",
    "deletion_betweenness_delta_rank",
    "deletion_betweenness_abs_delta_rank",
]


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def format_float(x):
    return format(x, '.17g')


def read_graph(edges_csv):
    graph=nx.Graph()
    names={}
    with edges_csv.open(newline='') as f:
        required={'protein1','protein2','combined_score'}
        r=csv.DictReader(f)
        if not required.issubset(r.fieldnames or []):
            raise ValueError(f'Edge CSV needs {required}. Got {r.fieldnames}')
        for row in r:
            if int(row['combined_score']) < THRESHOLD:
                continue
            a,b=row['protein1'],row['protein2']
            if a==b:continue
            graph.add_edge(a,b)
            if row.get('name1'):names[a]=row['name1']
            if row.get('name2'):names[b]=row['name2']
    if TARGET not in graph:raise ValueError('Target not found in the graph')
    comp=graph.subgraph(nx.node_connected_component(graph,TARGET)).copy()
    assert comp.degree(TARGET)==25, ('Wrong graph HLJ1 degree',comp.degree(TARGET))
    assert comp.number_of_nodes()==5716 and comp.number_of_edges()==104138, (
        'Graph differs from standardized functional 700 threshold',
        comp.number_of_nodes(),comp.number_of_edges())
    print(f'Input edges: {graph.number_of_edges():,}; HLJ1 component: '
          f'{len(comp):,} vertices, {comp.number_of_edges():,} edges')
    return comp,names


def compile_brandes(cpp_file, exe):
    """Compile normally, recovering missing Apple libc++ headers via the SDK."""
    cc = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if cc is None:
        raise RuntimeError('C++ compiler not found. Install Xcode Command Line Tools (macOS).')
    command = [cc, '-O3', '-std=c++17', '-pthread', str(cpp_file), '-o', str(exe)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode == 0:
        return

    # Some Apple CLT installations omit the compiler-relative libc++ headers,
    # while the selected SDK still supplies them. Do not retry unrelated errors.
    if sys.platform == 'darwin' and "'algorithm' file not found" in result.stderr:
        xcrun = shutil.which('xcrun')
        if xcrun:
            sdk = subprocess.run([xcrun, '--show-sdk-path'], text=True, capture_output=True)
            headers = Path(sdk.stdout.strip()) / 'usr/include/c++/v1'
            if sdk.returncode == 0 and (headers / 'algorithm').is_file():
                print(f'Retrying C++ compilation with SDK headers: {headers}', flush=True)
                retry = subprocess.run(
                    command + ['-isystem', str(headers)], text=True, capture_output=True,
                )
                if retry.returncode == 0:
                    return
                result = retry
    raise RuntimeError(f'Brandes C++ compilation failed:\n{result.stderr}')


def exact_betweenness(graph, nodes, cpp_file, workers):
    with tempfile.TemporaryDirectory() as tmp:
        exe=Path(tmp)/'brandes_exact'
        compile_brandes(cpp_file, exe)
        mapping={v:i for i,v in enumerate(nodes)}
        network=f'{len(nodes)} {graph.number_of_edges()}\n'+''.join(
            f'{mapping[u]} {mapping[v]}\n' for u,v in graph.edges())
        result=subprocess.run([str(exe),str(workers)],input=network,
                              text=True,capture_output=True,check=True)
    values=[float(x) for x in result.stdout.splitlines()]
    if len(values)!=len(nodes):raise ValueError('Unexpected Brandes output length')
    return dict(zip(nodes,values))


def compute_metrics(graph,names,cpp_file,workers):
    nodes=sorted(graph.nodes())
    orig=exact_betweenness(graph,nodes,cpp_file,workers)
    removed=graph.copy()
    removed.remove_node(TARGET)
    survivor=[v for v in nodes if v!=TARGET]
    post=exact_betweenness(removed,survivor,cpp_file,workers)
    hops=nx.single_source_shortest_path_length(graph,TARGET)
    out=[]
    for v in survivor:
        before,after=orig[v],post[v]
        delta=after-before
        ratio_after=delta/after if after!=0 else None
        ratio_before=delta/before if before!=0 else None
        orf=v.split('.',1)[1] if '.' in v else v
        row={
          'string_id':v, 'orf':orf if orf.startswith('Y') else '',
          'gene':names.get(v,orf),
          'deletion_degree_before':graph.degree(v),
          'deletion_degree_after':removed.degree(v),
          'deletion_distance_to_hlj1':hops[v],
          'deletion_betweenness_before':format_float(before),
          'deletion_betweenness_after':format_float(after),
          'deletion_betweenness_delta':format_float(delta),
          'deletion_betweenness_abs_delta':format_float(abs(delta)),
          'deletion_betweenness_ratio_to_after':format_float(ratio_after) if ratio_after is not None else '',
          'deletion_betweenness_ratio_to_before':format_float(ratio_before) if ratio_before is not None else '',
          'deletion_betweenness_pct_to_after':format_float(100*ratio_after) if ratio_after is not None else '',
          'deletion_betweenness_pct_to_before':format_float(100*ratio_before) if ratio_before is not None else '',
        }
        out.append(row)
    for col,out_col in [('deletion_betweenness_delta','deletion_betweenness_delta_rank'),
                        ('deletion_betweenness_abs_delta','deletion_betweenness_abs_delta_rank')]:
        ranks=sorted(out,key=lambda r:(-float(r[col]),r['string_id']))
        for rank,row in enumerate(ranks,1):row[out_col]=rank
    return out


def read_original_evidence(integrated_path):
    with integrated_path.open(newline='') as f:
        rd=csv.DictReader(f); fields=rd.fieldnames or []
        rows=list(rd)
    if 'string_id' not in fields:
        raise ValueError(f'Integrated evidence {integrated_path} is empty or missing string_id; '
                         'restore the original 4,439-candidate evidence CSV.')
    unique_ids = {r['string_id'] for r in rows}
    if len(rows) != 4439 or len(unique_ids) != 4439 or '' in unique_ids:
        raise ValueError(f'Integrated evidence {integrated_path}: expected 4,439 rows with '
                         f'unique nonempty STRING IDs, got {len(rows):,} rows and '
                         f'{len(unique_ids):,} distinct IDs.')
    renamed=[]
    for row in rows:
        row_new={}
        for col,val in row.items():
            if col in DROP_OLD:continue
            row_new[SEMANTIC_RENAMES.get(col,col)]=val
        renamed.append(row_new)
    final_fields=[SEMANTIC_RENAMES.get(c,c) for c in fields if c not in DROP_OLD]
    assert not any(('alex' in c.lower() or 'jadon' in c.lower() or 'junxi' in c.lower()) for c in final_fields)
    return final_fields,renamed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges',type=Path,default=PROJECT_ROOT/'outputs/jadon/string_v12.0_edges_tau0.7.csv')
    parser.add_argument('--integrated',type=Path,default=PROJECT_ROOT/'outputs/integrated/cross_method_evidence.csv')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/integrated')
    parser.add_argument('--workers',type=int,default=4)

    args=parser.parse_args()
    cpp_file=Path(__file__).with_name('brandes_exact.cpp')

    for path in [args.edges,args.integrated,cpp_file]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Validate merge inputs before spending time computing exact betweenness.
    fields, rows = read_original_evidence(args.integrated)
    graph, names = read_graph(args.edges)
    metrics = compute_metrics(graph, names, cpp_file, args.workers)
    standalone_fields = ['string_id', 'orf', 'gene'] + METRIC_COLS
    write_rows(args.output / 'deletion_betweenness_700.csv', standalone_fields, metrics)
    metric_map = {r['string_id'] : r for r in metrics}
    missing = [row['string_id'] for row in rows if row['string_id'] not in metric_map]

    if missing:
        raise ValueError(f'Unmatched candidate STRING IDs: {missing[:10]}')

    final_fields = fields + METRIC_COLS
    for row in rows:
        for col in METRIC_COLS:row[col] = metric_map[row['string_id']][col]

    write_rows(args.output / 'cross_method_evidence_700.csv', final_fields, rows)
    selected = {'APJ1','JJJ2','SSE1','UFD2','SSM4','FES1','MDJ1','ERJ5','YOS9','USA1','XDJ1','ERP5'}
    filtered = sorted([r for r in rows if r['gene'] in selected],key=lambda r:(
         ['Local','Cross','Boundary'].index(r['cadtp_track']),int(r['cadtp_rank_within_track'])))
    if len(filtered)!=len(selected):
        raise ValueError(f'Missing selected genes: {selected-set(r["gene"] for r in filtered)}')
    write_rows(args.output/'selected_protein_evidence_700.csv',final_fields,filtered)
    print(f'Wrote {len(metrics):,} deletion results; {len(rows):,} combined eligible proteins; '
          f'{len(filtered)} selected proteins to {args.output}')
    for row in filtered:
        if row['gene'] in ['APJ1','JJJ2','SSE1','UFD2','SSM4']:
            print(row['gene'], 'ΔB',row['deletion_betweenness_delta'],
                  'original-notebook post-relative',row['deletion_betweenness_ratio_to_after'])

if __name__=='__main__':main()
