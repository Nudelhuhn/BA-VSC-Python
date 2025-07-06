import os

class ReportGenerator:
    @staticmethod
    def generate_report(filenames, parent_dirs, labels, score_bins, output_path):
        output_path = os.path.join(output_path, "cluster_report.csv")
        grouped = {}
        for b, c, p_dir, f in zip(score_bins, labels, parent_dirs, filenames):  # group by score_bins and clusters
            if b not in grouped:
                grouped[b] = {}
            if c not in grouped[b]:
                grouped[b][c] = []
            grouped[b][c].append((p_dir, f))

        with open(output_path, "w", encoding="utf-8") as f: # write score_bins, clusters and file data to csv-file
            for b in sorted(grouped.keys()):
                f.write(f"Score-Bin: {b}\n")
                for c in sorted(grouped[b].keys()):
                    f.write(f"Cluster {c}:\n")
                    for (p_dir, subfolder), filename in grouped[b][c]:
                        f.write(f"  - {p_dir} - {subfolder} - {filename}\n")
                f.write("\n")
