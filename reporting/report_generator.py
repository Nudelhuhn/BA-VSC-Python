import os

class ReportGenerator:
    @staticmethod
    def generate_report(filenames, parent_dirs, labels, bins, output_path):
        output_path = os.path.join(output_path, "cluster_report.csv")
        grouped = {}
        for b, c, p_dir, f in zip(bins, labels, parent_dirs, filenames):
            if b not in grouped:
                grouped[b] = {}
            if c not in grouped[b]:
                grouped[b][c] = []
            grouped[b][c].append((p_dir, f))

        with open(output_path, "w", encoding="utf-8") as f:
            for b in sorted(grouped.keys()):
                f.write(f"Score-Bin: {b}\n")
                for c in sorted(grouped[b].keys()):
                    f.write(f"Cluster {c}:\n")
                    for (p_dir, subfolder), filename in grouped[b][c]:
                        f.write(f"  - {p_dir} - {subfolder} - {filename}\n")
                f.write("\n")
