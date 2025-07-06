class ScoreBinner:
    @staticmethod
    def bin_scores(scores, bins):
        binned_labels = []
        for score in scores:
            binned_label = None
            for low, high in bins:
                if low <= score <= high:
                    binned_label = f"{low}-{high}"  # label string for bin
                    break
            if binned_label is None:  # if no bin matched the score
                binned_label = "Unassigned"  # assign 'Unassigned' label
            binned_labels.append(binned_label)
        return binned_labels
