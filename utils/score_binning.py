class ScoreBinner:
    @staticmethod
    def bin_scores(scores, bins):  # Define a function to assign scores to bins
        binned_labels = []  # Initialize an empty list to store bin labels
        for score in scores:  # Loop through each score
            binned_label = None  # Initialize bin label as None for this score
            for low, high in bins:  # Loop through each defined bin range
                if low <= score <= high:  # Check if score falls within the current bin
                    binned_label = f"{low}-{high}"  # Create label string for this bin
                    break  # Exit bin loop once a match is found
            if binned_label is None:  # If no bin matched the score
                binned_label = "Unassigned"  # Assign 'Unassigned' label
            binned_labels.append(binned_label)  # Add the determined label to the list
        return binned_labels  # Return the list of bin labels
