def suggest_config_settings(num_files: int) -> dict:
    if num_files < 50:
        return {
            "dim_reduction": {
                "method": "umap",
                "params": {
                    "n_neighbors": 7,
                    "min_dist": 0.1,
                }
            },
            "clustering": {
                "method": "hdbscan",
                "params": {
                    "min_cluster_size": 2,
                    "min_samples": 1,
                }
            },
            "score_bins": [[0, 100]]
        }
    elif num_files < 300:
        return {
            "dim_reduction": {
                "method": "umap",
                "params": {
                    "n_neighbors": 20,
                    "min_dist": 0.2,
                }
            },
            "clustering": {
                "method": "hdbscan",
                "params": {
                    "min_cluster_size": 7,
                    "min_samples": 2,
                }
            },
            "score_bins": [[0, 49], [50, 79], [80, 89], [90, 94], [95, 99], [100, 100]]
        }
    else:  # 300 und mehr
        return {
            "dim_reduction": {
                "method": "umap",
                "params": {
                    "n_neighbors": 100,
                    "min_dist": 0.05,
                }
            },
            "clustering": {
                "method": "hdbscan",
                "params": {
                    "min_cluster_size": 15,
                    "min_samples": 7,
                    "cluster_selection_epsilon": 0.0,
                }
            },
            "score_bins": [[0, 49], [50, 54], [55, 59], [60, 64], [65, 69], [70, 74], [75, 79], [80, 84], [85, 89], [90, 94], [95, 99], [100, 100]]
        }
