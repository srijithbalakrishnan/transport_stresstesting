import osmnx as ox
import igraph as ig

# graph libraries
import networkx as nx
import numpy as np

# plotting
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
# tqdm already imported at the top of the file
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from shapely.geometry import LineString, MultiLineString, Point

import random
from pathlib import Path
import math
from libpysal.weights import W
from esda.moran import Moran
from scipy.spatial import distance_matrix
import pickle
import geopandas as gpd
from pyproj import CRS, Transformer
from functools import lru_cache
import shutil
import warnings


def _safe_divide(numer, denom):
    """Return numer/denom or 0.0 when denom is zero or not finite.

    Keeps behaviour deterministic and avoids ZeroDivisionError when an
    original metric equals zero.
    """
    try:
        if denom is None:
            return 0.0
        if denom == 0 or not np.isfinite(denom):
            return 0.0
        return numer / denom
    except Exception:
        return 0.0


def nx_to_igraph(G_nx):
    """Convert a NetworkX MultiDiGraph to an igraph Graph.

    This preserves node attributes and edge attributes (including keys) and
    attempts to keep node identifiers consistent between the two graph types.

    Parameters
    ----------
    G_nx : networkx.Graph or networkx.MultiDiGraph
        Input NetworkX graph.

    Returns
    -------
    ig.Graph
        An igraph Graph with node and edge attributes copied from ``G_nx``.
    """
    # Create igraph with same nodes and directionality
    G_ig = ig.Graph(directed=G_nx.is_directed())

    # Preserve original node IDs and attributes
    nodes = list(G_nx.nodes())
    G_ig.add_vertices(nodes)

    # Add node attributes if they exist
    if nodes and G_nx.nodes[nodes[0]]:
        for attr_name in G_nx.nodes[nodes[0]].keys():
            G_ig.vs[attr_name] = [G_nx.nodes[n].get(attr_name) for n in nodes]

    # Add edges with attributes
    edge_attrs_default = {
        "travel_time": None,
        "geometry": None,
        "osmid": None,
        "highway": None,
        "length": None,
        "key": 0,
    }

    # Prepare all edge attributes
    edge_data = []
    for u, v, key in G_nx.edges(keys=True):
        edge = {
            "source": u,
            "target": v,
            "key": key,
            **{
                attr: G_nx.edges[(u, v, key)].get(attr, default)
                for attr, default in edge_attrs_default.items()
            },
        }
        edge_data.append(edge)

    # Add edges with attributes
    G_ig.add_edges([(e["source"], e["target"]) for e in edge_data])

    # Set edge attributes
    for attr in edge_attrs_default:
        G_ig.es[attr] = [e[attr] for e in edge_data]

    return G_ig


def create_place_folder(place_names, base_dir="data/networks"):
    """Create (and return) a folder path for a list of place names.

    Parameters
    ----------
    place_names : str or list[str]
        Place name or list of place names used by OSMnx (e.g. "Thenkara, India").
    base_dir : str
        Base directory where networks are stored.

    Returns
    -------
    tuple
        (place_name, folder_path) where ``place_name`` is a cleaned string
        and ``folder_path`` is a pathlib.Path to the created directory.
    """

    country = place_names[0].split(",")[-1].strip()

    if isinstance(place_names, str):
        place_names = [place_names]

    base_names = []
    for name in place_names:
        base = name.split(",")[0].strip()
        clean_name = "".join(c if c.isalnum() or c == " " else "" for c in base)
        base_names.append(clean_name.replace(" ", "_"))

    folder_name = "_".join(base_names)
    place_name = "".join(base_names)
    folder_path = Path(base_dir) / country / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    return place_name, folder_path


def prepare_network(
    osmnx_place_list,
    network_type="drive",
    fallback_speed=30,
    base_dir="data/networks",
    force_redownload=False,
    only_results=False,
):
    """Download (or load cached) network(s) and convert to igraph.

    The function will attempt to load a cached igraph object if available;
    otherwise it downloads place graphs via OSMnx, projects them and
    converts them to igraph. Results and feature CSVs are saved under
    ``base_dir``.

    Parameters
    ----------
    osmnx_place_list : str or list[str]
        Place(s) to download via OSMnx.
    network_type : str
        OSMnx network type (e.g. 'drive').
    fallback_speed : int
        Fallback speed (km/h) used when adding edge speeds.
    base_dir : str
        Base directory for saving network data.
    force_redownload : bool
        If True, re-download even when cache exists.
    only_results : bool
        If True, only attempt to load existing results and skip saving.

    Returns
    -------
    (igraph.Graph, crs) or (None, None)
        The igraph object and its CRS, or (None, None) on failure.
    """

    print(f"Preparing network for: {osmnx_place_list}...")

    if isinstance(osmnx_place_list, str):
        osmnx_place_list = [osmnx_place_list]
    elif not isinstance(osmnx_place_list, (list, tuple)):
        raise ValueError("place_name must be string or list of strings")

    base_name, folder_path = create_place_folder(osmnx_place_list, base_dir=base_dir)
    pkl_path = folder_path / f"{base_name}.pkl"
    csv_path = folder_path / f"{base_name}_features.csv"
    results_path = folder_path / f"{base_name}_all.csv"

    if pkl_path.exists() and not force_redownload:
        print(f"......Loading cached igraph object from: {pkl_path}")
        if only_results:
            if results_path.exists():
                with open(pkl_path, "rb") as f:
                    G_ig = pickle.load(f)
                crs = None
            else:
                pass
        else:
            with open(pkl_path, "rb") as f:
                G_ig = pickle.load(f)
            crs = None
    else:
        G_nx = None
        random_dist = 2500  # np.random.randint(1000, 3000)
        for i, place in enumerate(osmnx_place_list):
            try:
                place_graph = ox.graph_from_place(
                    place, network_type=network_type, simplify=True, retain_all=False
                )
            except TypeError:
                center_point = ox.geocode(place)
                place_graph = ox.graph_from_point(
                    center_point,
                    dist=random_dist,
                    network_type=network_type,
                    simplify=True,
                    retain_all=False,
                )
                # select the largest connected component
            print(
                f"......Downloaded {place} graph with {len(place_graph.nodes())} nodes and {len(place_graph.edges())} edges"
            )
            place_graph = ox.add_edge_speeds(place_graph, fallback=fallback_speed)
            place_graph = ox.add_edge_travel_times(place_graph)
            place_graph = ox.project_graph(place_graph)

            if i == 0:
                G_nx = place_graph
            else:
                max_node = max(G_nx.nodes())
                mapping = {n: n + max_node + 1 for n in place_graph.nodes()}
                place_graph = nx.relabel_nodes(place_graph, mapping)
                G_nx = nx.compose(G_nx, place_graph)

        G_nx = nx.convert_node_labels_to_integers(
            G_nx, first_label=0, ordering="default", label_attribute="osmid"
        )

        # Convert to igraph
        G_ig = nx_to_igraph(G_nx)
        crs = G_nx.graph.get("crs", None)

        # Check edge count before saving
        num_edges = G_ig.ecount()
        if num_edges < 2520:
            # Save igraph object to Pickle
            with open(pkl_path, "wb") as f:
                pickle.dump(G_ig, f)
            print(f"......Saved igraph object to: {pkl_path}", crs)
        else:
            print(
                f"......Graph too large ({num_edges} edges), not saving. Deleting folder..."
            )
            try:
                shutil.rmtree(folder_path)
                print(f"......Deleted folder: {folder_path}")
            except Exception as e:
                print(f"......Error deleting folder {folder_path}: {e}")
            return None, None

    # get number of nodes and edges
    num_edges = G_ig.ecount()

    if num_edges < 2520:
        if only_results:
            if results_path.exists():
                print("......Extracting topological features...")
                features = extract_numeric_topological_features(G_ig, str(crs))
                features["place_name"] = base_name
                df = pd.DataFrame([features])
                df.to_csv(csv_path, index=False)
                print(f"......Saved features to: {csv_path}")
                return G_ig, crs
        else:
            print("......Extracting topological features...")
            features = extract_numeric_topological_features(G_ig, crs)
            features["place_name"] = base_name
            df = pd.DataFrame([features])
            df.to_csv(csv_path, index=False)
            print(f"......Saved features to: {csv_path}")
            return G_ig, crs
    else:
        print(f"......Graph too large ({num_edges} edges), skipping feature extraction")
        return None, None


def calculate_edge_importance(G, strategy, weight="travel_time"):
    """Compute edge importance metrics for an igraph graph.

    Parameters
    ----------
    G : igraph.Graph
        Input igraph graph.
    strategy : str
        One of 'betweenness','nearneighbour_edge','closeness','eigenvector',
        'degree','random' or 'all'.
    weight : str
        Edge attribute name to use as weight (default 'travel_time').

    Returns
    -------
    dict
        Mapping of strategy name to edge-score mapping.
    """

    if strategy == "all":
        all_metrics = {}

        strategies = [
            "betweenness",
            "nearneighbour_edge",
            "closeness",
            "eigenvector",
            "degree",
            # "pagerank",
            "random",
        ]

        for strat in strategies:
            try:
                metric_result = calculate_edge_importance(G, strat, weight)
                all_metrics.update(metric_result)
            except Exception as e:
                print(f"Warning: Failed to calculate {strat}: {str(e)}")
                continue

        return all_metrics

    if strategy == "betweenness":
        edge_betweenness = G.edge_betweenness(weights=weight)
        centrality = {
            "betweenness": {
                (v.source, v.target, v["key"]): edge_betweenness[v.index] for v in G.es
            }
        }
        return centrality

    elif strategy == "nearneighbour_edge":
        if G.is_weighted() and weight in G.edge_attributes():
            strength = G.strength(weights=weight)
            is_weighted = True
        else:
            degree = G.degree()
            is_weighted = False

        centrality = {"nearneighbour_edge": {}}

        for e in G.es:
            u = e.source
            v = e.target
            key = e["key"] if "key" in e.attributes() else 0

            if is_weighted:
                su = strength[u]
                sv = strength[v]
                w = e[weight] if weight in e.attributes() else 1.0
                numerator = su + sv - 2 * w
                denominator = abs(su - sv) + 1
                cn_e = (numerator / denominator) * w
            else:
                du = degree[u]
                dv = degree[v]
                numerator = du + dv - 2
                denominator = abs(du - dv) + 1
                cn_e = numerator / denominator

            centrality["nearneighbour_edge"][(u, v, key)] = cn_e
        return centrality

    elif strategy == "closeness":
        node_closeness = G.closeness(weights=weight)
        E = G.ecount()
        centrality = {"closeness": {}}

        for e in G.es:
            u = e.source
            v = e.target
            key = e["key"] if "key" in e.attributes() else 0

            ccu = node_closeness[u]
            ccv = node_closeness[v]
            denom = ccu + ccv
            if denom == 0:
                cce = 0
            else:
                cce = (E - 1) * (ccu * ccv) / denom
            centrality["closeness"][(u, v, key)] = cce

        return centrality

    elif strategy == "eigenvector":
        try:
            ev = G.eigenvector_centrality(weights=weight, scale=True)
        except Exception as e:
            raise ValueError(f"Eigenvector centrality failed: {str(e)}")

        centrality = {"eigenvector": {}}
        for e in G.es:
            u = e.source
            v = e.target
            key = e["key"] if "key" in e.attributes() else 0
            centrality["eigenvector"][(u, v, key)] = (ev[u] + ev[v]) / 2

        return centrality

    elif strategy == "degree":
        degrees = G.degree()
        centrality = {"degree": {}}
        for e in G.es:
            u = e.source
            v = e.target
            key = e["key"] if "key" in e.attributes() else 0
            centrality["degree"][(u, v, key)] = degrees[u] * degrees[v]
        return centrality

    # elif strategy == "pagerank":
    #     personalization = [G.degree(v) for v in G.vs]  # List of degrees for each node

    #     try:
    #         ppr = G.personalized_pagerank(reset=personalization, weights=weight)
    #     except Exception as e:
    #         raise ValueError(f"PPR computation failed: {str(e)}")

    #     centrality = {"pagerank": {}}
    #     for e in G.es:
    #         u = e.source
    #         v = e.target
    #         key = e["key"] if "key" in e.attributes() else 0
    #         centrality["pagerank"][(u, v, key)] = (ppr[u] + ppr[v]) / 2

    #     return centrality

    elif strategy == "random":
        centrality = {"random": {}}
        for e in G.es:
            key = e["key"] if "key" in e.attributes() else 0
            centrality["random"][
                (e.source, e.target, key)
            ] = random.random()  # Random float between 0 and 1
        return centrality

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def damage_network(G, damage_pct):
    """Randomly remove a percentage of edges from the graph.

    Parameters
    ----------
    G : igraph.Graph
        Input igraph graph.
    damage_pct : float
        Percentage (0-100) of edges to remove.

    Returns
    -------
    (igraph.Graph, list)
        A copy of the graph with edges removed and a list of removed edge tuples
        in the form (u, v, key).
    """
    if damage_pct < 0 or damage_pct > 100:
        raise ValueError("damage_pct must be between 0 and 100")

    # Store edges with their original attributes and keys
    edges = [(e.source, e.target, e["key"]) for e in G.es]

    # Calculate number of edges to remove
    remove_count = max(1, int(len(edges) * damage_pct / 100))
    rng = np.random.default_rng()
    edges_to_remove_idx = rng.choice(len(edges), remove_count, replace=False)
    edges_to_remove = [edges[i] for i in edges_to_remove_idx]

    # Create damaged copy
    damaged_G = G.copy()

    # Remove edges (using both node IDs and keys for accuracy)
    for u, v, key in edges_to_remove:
        eids = damaged_G.get_eids(
            [(u, v)], directed=damaged_G.is_directed(), error=False
        )

        # If multiple edges exist between same nodes, find the correct one by key
        for eid in eids:
            if damaged_G.es[eid]["key"] == key:
                damaged_G.delete_edges(eid)
                break

    return damaged_G, edges_to_remove


def restore_edges(G, damaged_G, metric_dict, removed_edges, restore_pct=1):
    """Restore a percentage of removed edges based on metric scores.

    Parameters
    ----------
    G : igraph.Graph
        Original graph.
    damaged_G : igraph.Graph
        Graph after damage.
    metric_dict : dict
        Mapping (u,v,key) -> score used to prioritise restoration.
    removed_edges : list
        List of removed edges as (u,v,key).
    restore_pct : float
        Percentage of candidate edges to restore.

    Returns
    -------
    igraph.Graph
        Graph with edges restored according to scores.
    """

    id_to_index = {v.index: v.index for v in G.vs}

    candidate_edges = {
        e: metric_dict.get(e, 0) for e in removed_edges if e in metric_dict
    }

    edges_sorted = sorted(candidate_edges.items(), key=lambda x: x[1], reverse=True)
    restore_count = int(len(candidate_edges) * restore_pct / 100)

    restored_G = damaged_G.copy()
    restored = 0

    for (u_id, v_id, _), _ in edges_sorted:
        if restored >= restore_count:
            break

        if u_id not in id_to_index or v_id not in id_to_index:
            continue
        u_idx = id_to_index[u_id]
        v_idx = id_to_index[v_id]

        if restored_G.are_connected(u_idx, v_idx):
            continue

        # Check if edge existed in the original graph
        eid = G.get_eid(u_idx, v_idx, directed=False, error=False)
        if eid == -1:
            continue  # edge not found in original

        restored_G.add_edge(
            u_idx,
            v_idx,
            **{k: v for k, v in G.es[eid].attributes().items() if k != "id"},
        )
        restored += 1

    return restored_G


def evaluate_network(
    disrupted_G,
    undisrupted_metrics,
    od_pairs=None,
    weight="travel_time",
    calculate_all_metrics=True,
):
    """Evaluate network performance after disruption.

    Parameters
    ----------
    disrupted_G : igraph.Graph
    undisrupted_metrics : dict
        Precomputed metrics from the undisrupted graph (closeness, coords).
    od_pairs : list of tuple, optional
        Origin-destination name pairs to compute efficiency for.
    weight : str
        Edge attribute used as weight for shortest paths.
    calculate_all_metrics : bool
        If True, compute Moran's I and clustering coefficients.

    Returns
    -------
    dict
        Dictionary containing 'efficiency', 'accessibility', 'morans_i', and
        'clustering_coefficient'.
    """

    # Precompute shortest paths once (directed, outward)
    shortest_paths = disrupted_G.shortest_paths(weights=weight, mode="out")
    normal_closeness = undisrupted_metrics["closeness"]
    node_coords = undisrupted_metrics["coords"]

    total_accessibility, node_accessibility = calculate_accessibility(
        disrupted_G,
        weight=weight,
        shortest_paths=shortest_paths,
        normal_closeness=normal_closeness,
    )

    # Reuse shortest_paths in other functions
    results = {
        "efficiency": calculate_efficiency(
            disrupted_G, od_pairs, weight, shortest_paths
        ),
        "accessibility": total_accessibility,
        "morans_i": None,
        "clustering_coefficient": None,
    }

    if calculate_all_metrics and node_coords is not None:
        accessibility = results["accessibility"]
        results["morans_i"] = calculate_morans_i(
            disrupted_G, node_accessibility, node_coords
        )

    if calculate_all_metrics:
        cc = directed_clustering_coefficient(disrupted_G)
        results["clustering_coefficient"] = sum(cc) / len(cc) if cc else 0

    return results


def calculate_efficiency(G, od_pairs=None, weight="travel_time", shortest_paths=None):
    """Compute total travel time over OD pairs or the full shortest-path matrix.

    This function sums shortest-path travel times. Without ``od_pairs`` it
    sums every finite origin->destination entry in ``shortest_paths``. With
    ``od_pairs`` it sums only the specified named pairs; vertex names are
    mapped via the ``name`` vertex attribute.

    Parameters
    ----------
    G : igraph.Graph
        Graph used when ``shortest_paths`` is None (to compute distances).
    od_pairs : list[tuple[str, str]] or None, optional
        Sequence of (origin_name, destination_name) pairs. Names must match
        the ``name`` vertex attribute of ``G``. Missing pairs are skipped.
    weight : str, optional
        Edge attribute to use as weight when computing shortest paths
        (default: ``'travel_time'``). Ignored if ``shortest_paths`` is given.
    shortest_paths : array-like or None, optional
        Precomputed (N, N) shortest-path matrix (rows=origins, cols=destinations).
        Can be a nested list or numpy array as returned by
        ``G.shortest_paths(weights=..., mode='out')``.

    Returns
    -------
    float
        Sum of finite shortest-path travel times for the requested pairs (or
        the entire matrix when ``od_pairs`` is None). Self-distances (0) and
        infinite distances are ignored.

    Notes
    -----
    - Missing names in ``od_pairs`` are silently skipped. The function
      returns 0.0 if no finite paths are found.
    """

    inf = float("inf")
    total_time = 0.0

    if shortest_paths is None:
        shortest_paths = G.shortest_paths(weights=weight, mode="out")

    if od_pairs is None:
        for row in shortest_paths:
            total_time += sum(t for t in row if t not in (0, inf))
    else:
        name_to_index = {v["name"]: v.index for v in G.vs if "name" in v.attributes()}
        for o_name, d_name in od_pairs:
            if o_name == d_name:
                continue
            try:
                o_idx = name_to_index[o_name]
                d_idx = name_to_index[d_name]
                path_time = shortest_paths[o_idx][d_idx]
                if path_time != inf:
                    total_time += path_time
            except KeyError:
                continue

    return total_time


def calculate_accessibility(
    G, weight="travel_time", shortest_paths=None, normal_closeness=None
):
    """Calculate accessibility for each node and total accessibility.

    Accessibility for node i is computed as:

    .. math::

        W_i = \sum_{j\neq i} \frac{C_j}{T_{ij}}

    where :math:`C_j` is the (undisrupted) closeness of node j and
    :math:`T_{ij}` is the travel-time from node i to node j.

    Parameters
    ----------
    G : igraph.Graph
        Graph used for indexing (vertex ordering). Must have a ``name``
        vertex attribute for mapping indices to node names when ``normal_closeness``
        is provided.
    weight : str, optional
        Edge attribute name used to compute shortest paths if ``shortest_paths``
        is None. Default: ``'travel_time'``.
    shortest_paths : array-like, optional
        Precomputed (N, N) travel-time matrix (rows=origins, cols=destinations).
        If not provided the function calls ``G.distances(weights=weight, mode='OUT')``.
    normal_closeness : dict
        Mapping of node name -> closeness value computed on the undisrupted graph.

    Returns
    -------
    total_accessibility : float
        Sum of per-node accessibility values for nodes with positive closeness.
    node_accessibility : dict
        Mapping node_name -> accessibility value (W_i).

    Notes
    -----
    - Nodes without a ``name`` attribute are skipped.
    - Entries where :math:`T_{ij}` is zero or infinite are ignored.
    - ``normal_closeness`` values that are NaN are filtered out.
    - The function raises ValueError if ``normal_closeness`` is None.
    """

    if shortest_paths is None:
        shortest_paths = G.distances(weights=weight, mode="OUT")

    if normal_closeness is None:
        raise ValueError("You must provide normal_closeness as a dictionary.")

    normal_closeness = {
        v["name"]: normal_closeness[v.index]
        for v in G.vs
        if "name" in v.attributes() and not math.isnan(normal_closeness[v.index])
    }

    name_to_index = {v.index: v["name"] for v in G.vs if "name" in v.attributes()}
    total_accessibility = 0.0
    node_accessibility = {}

    for i in range(G.vcount()):
        W_i = 0.0
        name_i = name_to_index.get(i)
        if name_i is None:
            continue  # Skip unnamed nodes

        for j, T_ij in enumerate(shortest_paths[i]):
            if i == j or T_ij in [0, float("inf")]:
                continue

            name_j = name_to_index.get(j)
            if name_j is None:
                continue

            closeness_j = normal_closeness.get(name_j, 0.0)
            W_i += closeness_j / T_ij

        node_accessibility[name_i] = W_i

        if normal_closeness.get(name_i, 0.0) > 0:
            total_accessibility += W_i

    return total_accessibility, node_accessibility


def calculate_morans_i(G, values, coords, alpha=1.0, threshold=1e-6):
    """Compute Moran's I statistic for node attribute values using inverse-distance weights.

    This constructs a spatial weights object where the weight between nodes i
    and j is 1 / (d_{ij}^alpha) (with a small ``threshold`` added to distances
    to avoid division by zero). The weights are converted into a libpysal
    weights object and Moran's I is computed via ``esda.Moran``.

    Parameters
    ----------
    G : igraph.Graph
        Graph used to determine vertex ordering and names. The function reads
        ``G.vs['name']``; ensure the ``name`` vertex attribute exists and its
        order corresponds to rows in ``coords``.
    values : dict
        Mapping from node name (matching ``G.vs['name']``) to numeric value.
    coords : array-like
        Numeric array-like of shape (N, 2) containing node coordinates. Rows
        must be in the same order as ``G.vs``.
    alpha : float, optional
        Power parameter for distance decay (default 1.0). Weights are
        computed as 1 / (distance**alpha).
    threshold : float, optional
        Small constant added to distances before inversion to avoid infinite
        weights for zero distances (default 1e-6).

    Returns
    -------
    float
        Moran's I statistic (float). Use the associated ``esda.Moran`` object
        for additional diagnostics (e.g., p-values) if needed.

    Notes
    -----
    - The function depends on ``scipy.spatial.distance_matrix``, ``libpysal``
      for weight construction and ``esda.Moran`` for the statistic.
    - If node names or coordinates are misaligned, the results will be
      incorrect; ensure consistent ordering between ``G.vs`` and ``coords``.
    """

    n = G.vcount()
    node_names = G.vs["name"]

    value_vector = np.array([values.get(name, 0.0) for name in node_names])

    euclidean_dist = distance_matrix(coords, coords)
    adjusted_distances = euclidean_dist + threshold

    weights_matrix = 1.0 / (adjusted_distances**alpha)
    np.fill_diagonal(weights_matrix, 0)

    neighbors = {}
    weights = {}
    for i in range(n):
        nonzero_indices = np.nonzero(weights_matrix[i])[0]
        neighbors[i] = list(nonzero_indices)
        weights[i] = list(weights_matrix[i][nonzero_indices])

    w = W(neighbors, weights)
    moran = Moran(value_vector, w)
    # print(f"Moran's I: {moran.I}, p-value: {moran.p_sim}")
    return moran.I


def directed_clustering_coefficient(G, mode="cycle"):
    """Compute directed clustering coefficient per node.

    Parameters
    ----------
    G : igraph.Graph
    mode : {'cycle','middleman','all'}
        Mode used for counting directed triangles.

    Returns
    -------
    list
        Per-node clustering coefficient values.
    """

    if not G.is_directed():
        raise ValueError("Graph must be directed.")

    valid_modes = {"cycle", "middleman", "all"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode. Choose from {valid_modes}")

    adj = {v.index: set(G.neighbors(v, mode="all")) for v in G.vs}
    cc = []

    for node in G.vs:
        u = node.index
        neighbors = list(adj[u])
        k = len(neighbors)

        if k < 2:
            cc.append(0.0)
            continue

        triangles = 0
        for i, a in enumerate(neighbors):
            for b in neighbors[i + 1 :]:
                if mode == "cycle":
                    if a in adj[b] and b in adj[u] and u in adj[a]:
                        triangles += 1
                elif mode == "middleman":
                    if a in adj[u] and b in adj[a] and b in adj[u]:
                        triangles += 1
                elif mode == "all":
                    if (a in adj[b] and b in adj[u] and u in adj[a]) or (
                        b in adj[a] and a in adj[u] and u in adj[b]
                    ):
                        triangles += 1

        max_possible = k * (k - 1)
        cc.append(triangles / max_possible)

    return cc


from tqdm import tqdm


def run_singlepct_experiment(
    G, strategy, damage_pct=10, iterations=50, restore_step=2, seed=None
):
    """Run repeated damage-and-restoration experiments for a single damage level.

    The function runs ``iterations`` trials. In each trial it randomly removes
    ``damage_pct`` percent of edges (via ``damage_network``), evaluates the
    degraded network and then simulates stepwise restoration according to the
    requested ``strategy``. Results from every restoration step are collected
    into a pandas DataFrame and returned concatenated across iterations.

    Parameters
    ----------
    G : igraph.Graph
        Original (undisrupted) igraph graph.
    strategy : str
        Restoration strategy name (e.g. 'betweenness', 'closeness', 'random' or
        'all' to evaluate all strategies).
    damage_pct : int
        Percentage of edges to remove for each disruption iteration.
    iterations : int
        Number of independent disruption iterations to run.
    restore_step : int
        Percentage of removed edges to restore at each restoration step.
    seed : int or None
        RNG seed to make experiments reproducible. If provided both Python's
        ``random`` and ``numpy.random`` are seeded.

    Returns
    -------
    pandas.DataFrame
        Concatenated results for all iterations. Key columns include:

        - ``disruption_iter``: disruption iteration index
        - ``restore_iter``: restoration step index
        - ``efficiency``: ratio (current / original) of efficiency metric
        - ``accessibility``: ratio (current / original) of accessibility
        - ``morans_i``: ratio (current / original) of Moran's I (0 if original is 0)
        - ``clustering``: ratio (current / original) of clustering coefficient
        - ``edges_restored``: cumulative edges restored at this step
        - ``edges_remaining``: remaining removed edges
        - ``strategy``: the restoration strategy used
        - ``disruption_pct`` and ``restore_pct``: metadata about experiment setup

    Notes
    -----
    - Many returned metrics are stored as ratios to the undisrupted (original)
      performance. If an original metric value is zero this may produce
      division-by-zero errors for some metrics or be handled as zero in others
      (see code for per-metric behavior). Ensure original metrics are
      non-zero when comparing ratios or handle zeros upstream.
    - When ``strategy`` is 'all', the function precomputes all edge metrics
      via ``calculate_edge_importance`` and iterates over them.
    """

    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)

    # Precompute all strategy metrics upfront
    edge_strategy_metrics = calculate_edge_importance(
        G, strategy="all" if strategy == "all" else strategy, weight="travel_time"
    )

    undisrupted_metrics = {}
    undisrupted_metrics["closeness"] = dict(
        zip(
            G.vs["name"],
            G.closeness(weights="travel_time", normalized=False, mode="out"),
        )
    )
    undisrupted_metrics["coords"] = np.array([[v["x"], v["y"]] for v in G.vs])

    original_perf = evaluate_network(G, undisrupted_metrics)
    all_results = []

    for iter_num in range(iterations):
        # Generate a single disruption for this iteration
        damaged_G, removed_edges = damage_network(G, damage_pct)
        current_perf = evaluate_network(damaged_G, undisrupted_metrics)

        # Determine which strategies to process
        strategies_to_process = (
            [strategy] if strategy != "all" else list(edge_strategy_metrics.keys())
        )

        for current_strategy in strategies_to_process:
            # Create a fresh copy of the damaged network for this strategy
            strategy_damaged_G = damaged_G.copy()
            strategy_removed_edges = removed_edges.copy()

            # Initialize results for this strategy
            iter_results = {
                "disruption_iter": [iter_num],
                "restore_iter": [0],
                "efficiency": [
                    _safe_divide(current_perf.get("efficiency", 0), original_perf.get("efficiency", 0))
                ],
                "accessibility": [
                    _safe_divide(current_perf.get("accessibility", 0), original_perf.get("accessibility", 0))
                ],
                "morans_i": [
                    _safe_divide(current_perf.get("morans_i", 0), original_perf.get("morans_i", 0))
                ],
                "clustering": [
                    _safe_divide(current_perf.get("clustering_coefficient", 0), original_perf.get("clustering_coefficient", 0))
                ],
                "edges_restored": [0],
                "edges_remaining": [len(strategy_removed_edges)],
                "strategy": [current_strategy],
            }
            if strategy == "eigenvector":
                print(iter_results["accessibility"], iter_results["morans_i"])

            edges_per_step = max(
                1, int(len(strategy_removed_edges) * restore_step / 100)
            )
            total_restore_steps = int(
                np.ceil(len(strategy_removed_edges) / edges_per_step)
            )
            restored_so_far = 0

            for restore_iter in range(1, total_restore_steps + 1):
                remaining_edges = len(strategy_removed_edges) - restored_so_far
                restore_pct_actual = (
                    restore_step
                    if remaining_edges > edges_per_step
                    else (remaining_edges / len(strategy_removed_edges)) * 100
                )

                restored_G = restore_edges(
                    G,
                    strategy_damaged_G,
                    edge_strategy_metrics[current_strategy],
                    strategy_removed_edges,
                    restore_pct_actual,
                )
                current_perf = evaluate_network(restored_G, undisrupted_metrics)
                strategy_damaged_G = restored_G

                newly_restored = min(edges_per_step, remaining_edges)
                restored_so_far += newly_restored

                # Append results for this restoration step
                iter_results["restore_iter"].append(restore_iter)
                iter_results["efficiency"].append(
                    min(1, _safe_divide(current_perf.get("efficiency", 0), original_perf.get("efficiency", 0)))
                )
                iter_results["accessibility"].append(
                    min(
                        1,
                        _safe_divide(current_perf.get("accessibility", 0), original_perf.get("accessibility", 0)),
                    )
                )
                iter_results["morans_i"].append(
                    min(1, _safe_divide(current_perf.get("morans_i", 0), original_perf.get("morans_i", 0)))
                )
                iter_results["clustering"].append(
                    min(
                        1,
                        _safe_divide(current_perf.get("clustering_coefficient", 0), original_perf.get("clustering_coefficient", 0)),
                    )
                )
                iter_results["edges_restored"].append(restored_so_far)
                iter_results["edges_remaining"].append(
                    len(strategy_removed_edges) - restored_so_far
                )
                iter_results["strategy"].append(current_strategy)

                if restored_so_far >= len(strategy_removed_edges):
                    break

            # Extend disruption_iter to match length of other arrays
            iter_results["disruption_iter"] = [iter_num] * len(
                iter_results["efficiency"]
            )

            # Create DataFrame for this iteration and strategy
            iter_df = pd.DataFrame(iter_results)
            iter_df["disruption_pct"] = damage_pct
            iter_df["restore_pct"] = restore_step
            all_results.append(iter_df)

    return pd.concat(all_results, ignore_index=True)


def run_multiple_damage_experiments(
    G,
    strategy,
    damage_pcts=range(10, 101, 10),
    iterations_per_damage=1,
    restore_pct=2,
    seed=None,
):
    """Run experiments across multiple damage percentage levels and collect results.

    This function iterates over the provided ``damage_pcts`` sequence and for
    each damage level calls :func:`run_singlepct_experiment` to run the
    restoration experiment. Results for each damage level are concatenated and
    returned as a single :class:`pandas.DataFrame`.

    Parameters
    ----------
    G : igraph.Graph or networkx.Graph
        The original (undisrupted) graph used for the experiments.
    strategy : str
        Restoration strategy name (e.g. ``'betweenness'``, ``'closeness'``,
        ``'random'``) or ``'all'`` to evaluate all strategies.
    damage_pcts : iterable of int, optional
        Sequence of damage percentage values to evaluate (default: ``range(10, 101, 10)``).
    iterations_per_damage : int, optional
        Number of independent disruption iterations to run for each damage level.
    restore_pct : int or float, optional
        Percentage of removed edges to restore per restoration step (passed to
        ``run_singlepct_experiment`` as ``restore_step``).
    seed : int or None, optional
        Random seed forwarded to ``run_singlepct_experiment`` for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame containing results across all damage levels.
        Each row corresponds to a single restoration step from a single
        disruption trial and typically contains columns such as:

        - ``damage_pct``: damage level for the block
        - ``strategy``: restoration strategy used
        - ``disruption_iter``: disruption trial index
        - ``restore_iter``: restoration step index
        - ``efficiency``, ``accessibility``, ``morans_i``, ``clustering``: performance metrics (ratios)
        - ``edges_restored``, ``edges_remaining``: restoration bookkeeping
        - ``restore_pct``, ``disruption_pct``: metadata about experimental setup

    Notes
    -----
    - If ``strategy`` is not ``'all'``, the returned DataFrame will have the
      ``strategy`` column set to the provided strategy for all rows.
    - Progress over damage levels is shown via :mod:`tqdm` when available.
    - The function concatenates DataFrames returned from
      :func:`run_singlepct_experiment`; any additional columns produced there
      will be included in the final result.
    """

    all_results = []

    for damage_pct in tqdm(damage_pcts, desc="Damage levels"):
        results_df = run_singlepct_experiment(
            G,
            strategy=strategy,
            damage_pct=damage_pct,
            iterations=iterations_per_damage,
            restore_step=restore_pct,
            seed=seed,
        )

        results_df["damage_pct"] = damage_pct
        # results_df["experiment_iteration"] = iteratioFprintn

        if strategy != "all":
            results_df["strategy"] = strategy

        all_results.append(results_df)

    combined_df = pd.concat(all_results, ignore_index=True)

    column_order = [
        "damage_pct",
        # "experiment_iteration",
        "strategy",
        "disruption_iter",
        "restore_iter",
        "edges_restored",
        "edges_remaining",
        "performance",
        "restore_pct",
        "disruption_pct",  # This is redundant with damage_pct but kept for backward compatibility
    ]

    existing_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[
        existing_columns
        + [col for col in combined_df.columns if col not in existing_columns]
    ]

    return combined_df


def plot_single_results(results_df, place_name, metric="efficiency"):
    """Plot restoration results from a single experiment DataFrame.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        The DataFrame containing the results of the restoration experiment.
    place_name : str
        The name of the place or network being analyzed.
    metric : str
        The performance metric to plot (e.g. ``'efficiency'``). If ``'all'``,
        all available metrics will be plotted.

    Returns
    -------
    None
        Displays a plot of the restoration results.
    
    Notes
    -----
    - The function uses seaborn and matplotlib for plotting.
    - The DataFrame is expected to contain columns such as ``'edges_restored'``,
      ``'strategy'``, and the specified performance metrics.
    """
    sns.set(style="whitegrid", context="paper", font_scale=1.4)
    available_metrics = ["efficiency", "accessibility", "morans_i", "clustering"]

    if metric != "all" and metric not in available_metrics:
        raise ValueError(f"Invalid metric: {metric}")

    fig, ax = plt.subplots(figsize=(8, 6))

    strategies = results_df["strategy"].unique()
    colors = sns.color_palette("Set1", n_colors=4)
    metrixc_dict = {
        "efficiency": "Efficiency",
        "accessibility": "Accessibility",
        "morans_i": "Spatial inequality",
        "clustering": "Connectivity",
    }

    if metric == "all":
        linestyles = {
            "efficiency": "solid",
            "accessibility": "dashed",
            "morans_i": "dashdot",
            "clustering": "dotted",
        }

        for j, m in enumerate(available_metrics):
            for i, strategy in enumerate(strategies):
                sns.lineplot(
                    data=results_df[results_df["strategy"] == strategy],
                    x="edges_restored",
                    y=m,
                    label=f"{metrixc_dict[m]}",
                    color=colors[j],
                    # linestyle=linestyles[m],
                    linewidth=2,
                    ax=ax,
                    errorbar=(
                        "ci",
                        95,
                    ),
                )
        ax.set_ylabel("Performance Metric")
        ax.set_xlim(
            0,
        )

    else:
        for i, strategy in enumerate(strategies):
            subset = results_df[results_df["strategy"] == strategy]
            grouped = subset.groupby("edges_restored")[metric].median().reset_index()
            sns.lineplot(
                data=results_df[results_df["strategy"] == strategy],
                x="edges_restored",
                y=metric,
                label=strategy,
                color=colors[i],
                linewidth=2,
                ax=ax,
                estimator="median",  # Optional: for median instead of mean
                errorbar=("ci", 95),  # Confidence interval
            )
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} (Ratio)")

    ax.set_xlabel("Edges Restored")
    ax.set_ylim(0, 1.05)
    # only left and bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # black color for left and bottom spines
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.set_title(
        f"Network recovery based on {strategy} strategy\n{place_name}",
        pad=20,
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Performance metric", loc="lower right")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from shapely.geometry import LineString, MultiLineString
import contextily as ctx
import numpy as np
import seaborn as sns
import pyproj


def plot_world_location(G, crs, figsize=(5, 1), zoom=5):  # Increased zoom
    """Plot a world map with an inset showing the location of the study area."""

    fig, inset_ax = plt.subplots()
    try:
        all_x = [v["x"] for v in G.vs]
        all_y = [v["y"] for v in G.vs]
        centroid_x = np.mean(all_x)
        centroid_y = np.mean(all_y)
        #convert centroid to epsg:4326
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        centroid_x, centroid_y = transformer.transform(centroid_x, centroid_y)
    
        
        # Make inset background semi-transparent
        inset_ax.patch.set_alpha(0.85)  # Semi-transparent background
        inset_ax.set_facecolor('white')  # White background with transparency
        
        # Use contextily for world map with low zoom
        try:
            # Set world bounds
            world_bounds = [-180, -90, 180, 90]
            inset_ax.set_xlim(world_bounds[0], world_bounds[2])
            inset_ax.set_ylim(world_bounds[1], world_bounds[3])
            
            # Add world basemap with very low zoom
            ctx.add_basemap(inset_ax, crs='EPSG:4326', zoom=1,
                            source=ctx.providers.Esri.WorldStreetMap,
                            alpha=1, attribution="© Esri, contributors, etc.")  # Make basemap slightly transparent
            
        except Exception as basemap_error:
            # Fallback: create a simple world outline
            self._create_simple_world_outline(inset_ax)
            inset_ax.set_facecolor('white')
            inset_ax.patch.set_alpha(0.9)
        
        # Plot centroid with emphasis (higher zorder)
        inset_ax.scatter(centroid_x, centroid_y, color='red', s=10, 
                        zorder=10, marker='*', edgecolor='red', linewidth=0.5)
        
        # Add study area indicator (rectangle or circle)
        study_buffer = 10
        rect = plt.Rectangle((centroid_x - study_buffer/2, centroid_y - study_buffer/2),
                            study_buffer, study_buffer,
                            fill=False, color='red', linewidth=1, linestyle='-',
                            alpha=0.8, zorder=5)
        inset_ax.add_patch(rect)
        
        # Style the inset with borders and shadow effect
        # inset_ax.set_title('Global Location', fontsize=9, pad=4, weight='bold')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        
        # Add border to make inset stand out
        for spine in inset_ax.spines.values():
            spine.set_visible(False)
            spine.set_color('gray')
            spine.set_linewidth(1)
            # spine.set_zorder(10)
            
    except Exception as e:
        print(f"Could not create location inset: {e}")
    return fig, inset_ax


def plot_network_with_edge_attribute(
    G,
    crs,
    attribute_name="travel_time",
    cmap="RdBu_r",
    node_size=0,
    edge_linewidth=1.25,
    colorbar_label=None,
    figsize=(10, 10),
    show=True,
    legend=True,
    close=False,
    use_geometry=True,
):
    """Plot an igraph network with edges colored by a specified attribute.

    Parameters
    ----------
    G : igraph.Graph
        The igraph graph to plot. Must have 'x' and 'y' vertex attributes
        for coordinates, and the specified edge attribute.
    crs : dict
        Coordinate reference system for basemap (e.g., {'init': 'epsg:4326'}).
    attribute_name : str
        Edge attribute name to color by (default: 'travel_time').
    cmap : str
        Matplotlib colormap name (default: 'RdBu_r').
    node_size : int
        Size of nodes to plot (default: 0, meaning no nodes).
    edge_linewidth : float
        Line width for edges (default: 1.25).
    colorbar_label : str or None
        Label for the colorbar (default: None).
    figsize : tuple
        Figure size (default: (10, 10)).
    show : bool
        Whether to display the plot (default: True).
    legend : bool
        Whether to include a colorbar legend (default: True).
    close : bool
        Whether to close the plot after showing (default: False).
    use_geometry : bool
        Whether to use edge 'geometry' attribute if available (default: True).
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects of the plot.
    
    Notes
    -----
    - Requires geopandas, contextily, matplotlib, seaborn, and shapely.
    - If the specified edge attribute is missing, raises ValueError.
    - If vertex coordinates are missing, attempts to use edge geometries.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_context("paper", font_scale=1.4)

    if attribute_name not in G.edge_attributes():
        available = G.edge_attributes()
        plt.close()
        raise ValueError(
            f"Attribute '{attribute_name}' not found. Available edge attributes: {available}"
        )

    if not all(attr in G.vertex_attributes() for attr in ["x", "y"]):
        print("Warning: Missing vertex coordinates - attempting to use geometry only")

    edge_attrs = [e[attribute_name] for e in G.es if attribute_name in e.attributes()]
    if not edge_attrs:
        plt.close()
        raise ValueError(f"No edges have attribute '{attribute_name}'")

    norm = Normalize(vmin=min(edge_attrs), vmax=max(edge_attrs))
    cmap = get_cmap(cmap)

    for edge in G.es:
        attr_value = edge[attribute_name]
        color = cmap(norm(attr_value)) if attr_value is not None else (0.5, 0.5, 0.5, 1)

        if use_geometry and "geometry" in edge.attributes():
            geom = edge["geometry"]
            try:
                if isinstance(geom, LineString) or (
                    isinstance(geom, MultiLineString) and not geom.is_empty
                ):
                    x, y = geom.xy
                    ax.plot(
                        x,
                        y,
                        color=color,
                        linewidth=edge_linewidth,
                    )
                    continue
            except Exception as e:
                print(f"Edge {edge.index} geometry error: {str(e)}")

        if all(attr in G.vertex_attributes() for attr in ["x", "y"]):
            u, v = edge.source, edge.target
            xu, yu = G.vs[u]["x"], G.vs[u]["y"]
            xv, yv = G.vs[v]["x"], G.vs[v]["y"]
            ax.plot(
                [xu, xv],
                [yu, yv],
                color=color,
                linewidth=edge_linewidth,
                solid_capstyle="round",
            )
        else:
            print(
                f"Warning: Edge {edge.index} - No geometry or coordinates available for plotting"
            )
            continue

    # Add nodes if requested
    if node_size > 0 and all(attr in G.vertex_attributes() for attr in ["x", "y"]):
        xs = [v["x"] for v in G.vs]
        ys = [v["y"] for v in G.vs]
        ax.scatter(xs, ys, s=node_size, color="black", zorder=2)

    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.3)  # pad adjusts spacing
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal", shrink=0.6,
        aspect = 20)
        cbar.set_label(colorbar_label, loc="center")

    if all(attr in G.vertex_attributes() for attr in ["x", "y"]):
        ax.set_aspect("equal")
        all_x = [v["x"] for v in G.vs]
        all_y = [v["y"] for v in G.vs]
        offsets = 500, 300
        # if all_x difference between min_max < threshold, then width must be at least threshold,
        #y can be based on current range
        threshold = 5000
        if max(all_x) - min(all_x) < threshold:
            x_offsets = threshold / 2
        else:
            x_offsets = offsets[0]
        y_offsets = offsets[1]
        ax.set_xlim(min(all_x) - x_offsets, max(all_x) + x_offsets)
        ax.set_ylim(min(all_y) - y_offsets, max(all_y) + y_offsets)

    ax.axes.set_axis_off()

    ctx.add_basemap(
        ax,
        crs=crs,
        source=ctx.providers.CartoDB.Positron,
        # zoom=15,
        alpha=1,
        attribution=False,
    )

    # ax.annotate(
    #     f"Network: Thenkara, India",
    #     xy=(0.98, 0.05),
    #     xycoords="axes fraction",
    #     ha="right",
    #     va="bottom",
    #     fontsize=11,
    #     color="black",
    # )

    return fig, ax


# def plot_network_with_edge_attribute(
#     G,
#     crs,
#     attribute_name="travel_time",
#     cmap="RdBu_r",
#     node_size=0,
#     edge_linewidth=1.25,
#     colorbar_label=None,
#     figsize=(10, 8),
#     show=True,
#     legend=True,
#     close=False,
#     use_geometry=True,
#     show_location_map=True,
#     height_ratios=(1, 6),   # world map : network map
#     width_ratios=(3, 1, 1), # center the top map in 3-column grid
# ):
#     """
#     Plot an igraph network with edges colored by a specified attribute,
#     using GridSpec layout where the world map is smaller and centered above the main map.
#     """
#     sns.set_context("paper", font_scale=1.3)
#     fig = plt.figure(figsize=figsize)

#     # Two rows, three columns: [empty, worldmap, empty] / [network spanning all]
#     gs = gridspec.GridSpec(
#         2, 3, height_ratios=height_ratios, width_ratios=width_ratios, hspace=0.05, wspace=0.0, figure=fig
#     )

#     # --- WORLD MAP (center column of top row) ---
#     if show_location_map and all(attr in G.vertex_attributes() for attr in ["x", "y"]):
#         world_ax = fig.add_subplot(gs[0, 0])  # middle column
#         world_ax.set_facecolor("white")

#         # Compute centroid in lat/lon
#         project = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
#         all_x = [v["x"] for v in G.vs]
#         all_y = [v["y"] for v in G.vs]
#         centroid_x, centroid_y = np.mean(all_x), np.mean(all_y)
#         centroid_x, centroid_y = project(centroid_x, centroid_y)

#         # Add low-zoom world basemap
#         try:
#             ctx.add_basemap(
#                 world_ax,
#                 crs="EPSG:4326",
#                 source=ctx.providers.OpenStreetMap.Mapnik,
#                 zoom=1,
#                 attribution=False,
#             )
#         except Exception as e:
#             print(f"World basemap error: {e}")
#             world_ax.set_facecolor("lightblue")

#         # Plot centroid and bounding box
#         world_ax.scatter(centroid_x, centroid_y, color="red", s=70, zorder=5, marker="*")
#         rect_size_x, rect_size_y = 30, 20
#         rect = plt.Rectangle(
#             (centroid_x - rect_size_x / 2, centroid_y - rect_size_y / 2),
#             rect_size_x,
#             rect_size_y,
#             linewidth=1.2,
#             edgecolor="red",
#             facecolor="none",
#         )
#         world_ax.add_patch(rect)

#         world_ax.set_xlim(-180, 180)
#         world_ax.set_ylim(-90, 90)
#         world_ax.set_xticks([])
#         world_ax.set_yticks([])
#         world_ax.set_title("Global Location", fontsize=12)

#     # --- NETWORK MAP (bottom row, full width) ---
#     ax = fig.add_subplot(gs[1, :])  # span all 3 columns

#     # Edge colors
#     edge_attrs = [e[attribute_name] for e in G.es if attribute_name in e.attributes()]
#     if not edge_attrs:
#         plt.close()
#         raise ValueError(f"No edges have attribute '{attribute_name}'")

#     norm = Normalize(vmin=min(edge_attrs), vmax=max(edge_attrs))
#     cmap = get_cmap(cmap)

#     # Draw edges
#     for edge in G.es:
#         attr_value = edge[attribute_name]
#         color = cmap(norm(attr_value)) if attr_value is not None else (0.5, 0.5, 0.5, 1)
#         if use_geometry and "geometry" in edge.attributes():
#             geom = edge["geometry"]
#             try:
#                 if isinstance(geom, LineString) or (
#                     isinstance(geom, MultiLineString) and not geom.is_empty
#                 ):
#                     x, y = geom.xy
#                     ax.plot(x, y, color=color, linewidth=edge_linewidth)
#                     continue
#             except Exception as e:
#                 print(f"Geometry error: {e}")
#         u, v = edge.source, edge.target
#         xu, yu = G.vs[u]["x"], G.vs[u]["y"]
#         xv, yv = G.vs[v]["x"], G.vs[v]["y"]
#         ax.plot([xu, xv], [yu, yv], color=color, linewidth=edge_linewidth)

#     # Nodes
#     if node_size > 0:
#         xs = [v["x"] for v in G.vs]
#         ys = [v["y"] for v in G.vs]
#         ax.scatter(xs, ys, s=node_size, color="black", zorder=2)

#     # Colorbar
#     if legend:
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         sm._A = []
#         cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
#         cbar.set_label(colorbar_label or attribute_name, fontsize=11)

#     # Map extent & basemap
#     all_x = [v["x"] for v in G.vs]
#     all_y = [v["y"] for v in G.vs]
#     ax.set_aspect("equal")
#     ax.set_axis_off()
#     offsets = 500, 300
#     ax.set_xlim(min(all_x) - offsets[0], max(all_x) + offsets[0])
#     ax.set_ylim(min(all_y) - offsets[1], max(all_y) + offsets[1])
#     ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.Positron, attribution=False)

#     ax.set_title("Network Map", fontsize=12)

#     if show:
#         plt.show()
#     if close:
#         plt.close(fig)

#     return fig, ax


def plot_restoration_across_strategies(
    df, damage_pct=90, place_name="Piedmont", ax=None, figsize=None
):
    """Plot restoration results across strategies for a given damage percentage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing restoration results with columns:
        'damage_pct', 'place_name', 'restore_pct_status', 'efficiency', 'strategy'.
    damage_pct : int
        Damage percentage to filter results (default: 90).
    place_name : str
        Name of the place/network to filter results (default: 'Piedmont').
    ax : matplotlib Axes or None
        Axes to plot on. If None, a new figure and axes are created (default: None).
    figsize : tuple or None
        Figure size if creating a new figure (default: None, which uses (8, 6)).
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects of the plot.
    
    Notes
    -----
    - The function uses seaborn and matplotlib for plotting.
    - The DataFrame is expected to contain the specified columns.
    """
    sns.set(style="whitegrid", context="paper", font_scale=1.3)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    plot_data = df[df["damage_pct"] == damage_pct]
    plot_data = plot_data[plot_data["place_name"] == place_name]

    plot_data["strategy"] = plot_data["strategy"].replace(
        {
            "betweenness": "Edge Betweenness",
            "nearneighbour_edge": "Nearest neighbour edge",
            "closeness": "Closeness",
            "eigenvector": "Eigenvector",
            "degree": "Degree",
            "pagerank": "PageRank",
            "random": "Random",
        }
    )

    sns.lineplot(
        data=plot_data,
        x="restore_pct_status",
        y="efficiency",
        hue="strategy",
        ax=ax,
    )

    ax.legend(title="Restoration Strategy", loc="upper left")
    ax.set_xlabel("Restored links (%)")
    ax.set_ylabel("Operational efficiency")

    ax.annotate(
        f"Initial damage:\n{damage_pct}% links",
        xy=(0.98, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=12,
        color="dimgray",
    )
    # ax.annotate(
    #     f"{place_name}, India",
    #     xy=(0.98, 0.1),
    #     xycoords="axes fraction",
    #     ha="right",
    #     va="bottom",
    #     fontsize=11,
    #     color="black",
    # )

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_facecolor("white")

    return fig, ax


def plot_combined_results(combined_df, G, strategy):
    """Plot combined restoration results across damage levels.

    Parameters
    ----------
    combined_df : pandas.DataFrame
        DataFrame containing combined restoration results with columns:
        'damage_pct', 'edges_restored', 'performance'.
    G : igraph.Graph or networkx.Graph
        The original (undisrupted) graph used for the experiments.
    strategy : str
        Restoration strategy name (e.g. 'betweenness', 'closeness', 'random').
    
    Returns
    -------
    None
        Displays a plot of the combined restoration results.
    
    Notes
    -----
    - The function uses seaborn and matplotlib for plotting.
    - The DataFrame is expected to contain the specified columns.
    """
    sns.set(style="ticks", context="paper", font_scale=1.2)
    plt.figure(figsize=(9, 5))
    damage_levels = sorted(combined_df["damage_pct"].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(damage_levels)))

    for i, damage_pct in enumerate(damage_levels):
        subset = combined_df[combined_df["damage_pct"] == damage_pct]
        sns.lineplot(
            data=subset,
            x="edges_restored",
            y="performance",
            color=colors[i],
            label=f"{damage_pct}% Damage",
            alpha=0.9,
            linewidth=2,
            estimator="median",
            errorbar=("ci", 95),
        )

    plt.title("Network Recovery Across Damage Levels", pad=20)
    plt.xlabel("Number of edges recovered", labelpad=10)
    plt.ylabel("Normalized Performance", labelpad=10)
    plt.ylim(0, 1.05)

    # Get total edges count
    if isinstance(G, nx.Graph):
        total_edges = G.number_of_edges()
    else:
        total_edges = G.ecount()

    plt.annotate(
        f"Total Edges: {total_edges}\nRecovery strategy: {strategy}",
        xy=(0.98, 0.2),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Initial damage", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def collapse_multigraph_to_graph_with_mapping(G, weight="weight"):
    """
    Collapse a multigraph (NetworkX or igraph) into a simple graph by merging
    parallel edges and summing their weights. Also returns a mapping from the
    edges in the new graph back to the original multigraph edges.

    Parameters
    ----------
    G : networkx.MultiGraph or igraph.Graph
        The input multigraph to collapse.
    weight : str
        The edge attribute to use as weight. Default is 'weight'.
    
    Returns
    -------
    H : networkx.Graph or igraph.Graph
        The collapsed simple graph.
    edge_mapping : dict
        A mapping from edges in H to lists of original edges in G.
    
    Notes
    -----
    - For NetworkX, the function handles undirected multigraphs.
    - For igraph, the function assumes the input graph is undirected and treats
      all edges as parallel edges between their endpoints.
    - If the specified weight attribute is missing on an edge, a default weight
      of 1 is used.
    """
    if isinstance(G, nx.Graph):
        # NetworkX case
        H = nx.Graph()
        edge_mapping = {}  # (u,v) -> list of (u,v,k)

        for u, v, k, data in G.edges(data=True):
            edge_key = tuple(sorted((u, v)))  # undirected
            w = data.get(weight, 1)

            if H.has_edge(*edge_key):
                H[edge_key[0]][edge_key[1]][weight] += w
            else:
                H.add_edge(*edge_key, **{weight: w})

            edge_mapping.setdefault(edge_key, []).append((u, v, k))
    else:
        # igraph case (treated as multigraph)
        H = ig.Graph()
        edge_mapping = {}

        # Add all vertices first
        H.add_vertices(len(G.vs))

        # Create a dictionary to map original vertex names to new indices
        name_to_index = {v["name"]: v.index for v in G.vs}

        # Process edges
        edge_weights = {}
        for e in G.es:
            u_name = G.vs[e.source]["name"]
            v_name = G.vs[e.target]["name"]
            edge_key = tuple(sorted((u_name, v_name)))
            w = e[weight] if weight in e.attributes() else 1

            if edge_key in edge_weights:
                edge_weights[edge_key] += w
            else:
                edge_weights[edge_key] = w

            edge_mapping.setdefault(edge_key, []).append(
                (u_name, v_name, 0)
            )  # Using 0 as key

        # Add edges to the new graph
        for edge_key, w in edge_weights.items():
            u_name, v_name = edge_key
            u = name_to_index[u_name]
            v = name_to_index[v_name]
            H.add_edge(u, v, **{weight: w})

        # Add vertex names back
        for v in G.vs:
            H.vs[name_to_index[v["name"]]]["name"] = v["name"]

    return H, edge_mapping


def extract_numeric_topological_features(g, crs):
    """Extract numeric topological features from an igraph graph.

    Parameters
    ----------
    g : igraph.Graph
        The input igraph graph.
    crs : dict
        Coordinate reference system of the graph (e.g., {'init': 'epsg:4326'}).
    
    Returns
    -------
    dict
        A dictionary of extracted topological features.

    Notes
    -----
    - The function computes various graph metrics including number of nodes,
      edges, density, average degree, degree standard deviation, connectivity,
      clustering coefficients, assortativity, and edge attribute summaries if
      present.
    - If edge attributes 'length' or 'travel_time' are present, their total and
      average values are included.
    """
    features = {}

    # Basic structure
    features["n_nodes"] = g.vcount()
    features["n_edges"] = g.ecount()
    features["density"] = g.density()
    features["avg_degree"] = sum(g.degree()) / g.vcount() if g.vcount() else 0

    # Degree variance
    degrees = g.degree()
    if degrees:
        mean_deg = features["avg_degree"]
        features["degree_stddev"] = (
            sum((d - mean_deg) ** 2 for d in degrees) / len(degrees)
        ) ** 0.5
    else:
        features["degree_stddev"] = 0

    is_conn = g.is_connected()

    features["global_clustering"] = g.transitivity_undirected()
    features["avg_clustering"] = g.transitivity_avglocal_undirected()

    features["assortativity_degree"] = g.assortativity_degree(directed=g.is_directed())

    if "length" in g.edge_attributes():
        lengths = g.es["length"]
        features["total_length"] = sum(lengths)
        features["avg_length"] = sum(lengths) / len(lengths) if lengths else 0

    if "travel_time" in g.edge_attributes():
        times = g.es["travel_time"]
        features["total_travel_time"] = sum(times)
        features["avg_travel_time"] = sum(times) / len(times) if times else 0
        features["crs"] = crs

    features["longitude"], features["latitude"] = sum(g.vs["x"]) / len(g.vs), sum(
        g.vs["y"]
    ) / len(g.vs)

    return features


def convert_to_target_crs(
    df,
    target_crs="EPSG:4326",
    lon_col="longitude",
    lat_col="latitude",
    crs_col="crs",
    drop_invalid=True,
):
    """Convert a table of coordinates (or a GeoDataFrame) to a target CRS.

    The function accepts a pandas.DataFrame or a geopandas.GeoDataFrame. If a
    GeoDataFrame with a valid ``.crs`` is provided the fast-path
    ``GeoDataFrame.to_crs`` is used. Otherwise the function expects columns
    containing longitude, latitude and a per-row source CRS and will transform
    points to ``target_crs`` producing a GeoDataFrame with a ``geometry``
    column.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Input table containing coordinate columns or a GeoDataFrame.
    target_crs : str or pyproj.CRS, optional
        Target coordinate reference system (default: ``'EPSG:4326'``).
    lon_col : str, optional
        Column name containing the X coordinate (default: ``'longitude'``).
    lat_col : str, optional
        Column name containing the Y coordinate (default: ``'latitude'``).
    crs_col : str, optional
        Column name containing the source CRS for each row. Values can be
        strings or objects accepted by ``pyproj.CRS.from_user_input``.
    drop_invalid : bool, optional
        If True drop rows for which a valid point cannot be produced
        (missing coords, unknown CRS, transform failure). If False the row
        will have a ``None`` geometry entry. Default: True.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with geometry column in ``target_crs``.

    Notes
    -----
    - The function will batch-transform rows grouped by identical source CRS
      strings to avoid repeated transformer construction and to be faster than
      transforming row-by-row with a fresh transformer each time.
    - For GeoDataFrame inputs with an existing ``geometry`` and ``crs`` the
      built-in ``to_crs`` is preferred and used when possible.
    """

    target_crs_obj = CRS.from_user_input(target_crs)

    # Fast path: already a GeoDataFrame with geometry and CRS
    if isinstance(df, gpd.GeoDataFrame):
        if df.crs is not None:
            try:
                return df.to_crs(target_crs_obj)
            except Exception as e:
                warnings.warn(f"GeoDataFrame.to_crs failed, falling back to row-wise transform: {e}")

        # If geometry missing but lon/lat present, fall through to tabular path
        if "geometry" in df.columns and df.crs is None:
            # continue to tabular path below
            pass

    # Tabular path: ensure required columns exist
    if not all(col in df.columns for col in [lon_col, lat_col, crs_col]):
        raise ValueError(f"Missing required columns ({lon_col}, {lat_col}, {crs_col}).")

    @lru_cache(maxsize=128)
    def get_transformer(src_crs_str):
        src = CRS.from_user_input(src_crs_str)
        return Transformer.from_crs(src, target_crs_obj, always_xy=True), src

    out_geoms = []

    # Group by unique CRS to reduce transformer constructions and use vector
    # transforms where possible.
    df_copy = df.copy()
    unique_crs_values = df_copy[crs_col].fillna("").unique()

    for src_crs_val in unique_crs_values:
        mask = df_copy[crs_col].fillna("") == src_crs_val
        sub = df_copy.loc[mask, [lon_col, lat_col]]

        # Prepare output list for this block
        block_points = [None] * len(sub)

        if sub.empty:
            out_geoms.extend(block_points)
            continue

        # If coordinates are missing entirely, fill with None/NaN
        xs = sub[lon_col].to_numpy()
        ys = sub[lat_col].to_numpy()

        try:
            transformer, src_crs = get_transformer(str(src_crs_val))
        except Exception as e:
            warnings.warn(f"Invalid source CRS '{src_crs_val}': {e}")
            out_geoms.extend([None] * len(sub))
            continue

        # If the source CRS is equal to the target, no transform required
        same_crs = src_crs == target_crs_obj

        for i, (x, y) in enumerate(zip(xs, ys)):
            if pd.isna(x) or pd.isna(y):
                block_points[i] = None
                continue

            if same_crs:
                try:
                    block_points[i] = Point(x, y)
                except Exception:
                    block_points[i] = None
            else:
                try:
                    x_new, y_new = transformer.transform(x, y)
                    block_points[i] = Point(x_new, y_new)
                except Exception as e:
                    warnings.warn(f"Coordinate transform failed for ({x}, {y}): {e}")
                    block_points[i] = None

        out_geoms.extend(block_points)

    df_copy["geometry"] = out_geoms

    gdf = gpd.GeoDataFrame(df_copy, geometry="geometry", crs=target_crs)

    if drop_invalid:
        gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)

    return gdf
