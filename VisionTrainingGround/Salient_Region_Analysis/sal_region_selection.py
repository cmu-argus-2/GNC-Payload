import pandas as pd

# Helper function to check if two MGRS labels share an edge or corner
def is_adjacent(label1, label2):
    # Extract number and letter from the MGRS labels
    num1, letter1 = int(label1[:-1]), label1[-1]
    num2, letter2 = int(label2[:-1]), label2[-1]

    valid_letters = "CDEFGHJKLMNPQRSTUVWX"
    
    # Ensure the letters are valid (They always will be, this is just a safety)
    if letter1 not in valid_letters or letter2 not in valid_letters:
        return True
    
    # Check adjacency for numbers and letters
    if abs(num1 - num2) <= 1 and abs(valid_letters.index(letter1) - valid_letters.index(letter2)) <= 1:
        return True
    return False


# Function to select labels based on value and adjacency conditions
def select_mgrs_labels(csv_file, n=None):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract the MGRS number
    df['Number'] = df['Label'].str.extract('(\d+)')

    # Sort by Value in descending order to easily pick the highest first
    df = df.sort_values(by='Value', ascending=False)

    # Initialize result list and a set for picked numbers
    result = []
    picked_numbers = set()

    # Pick the highest value and add it to the result
    result.append(df.iloc[0])
    picked_numbers.add(df.iloc[0]['Label'][:-1])  # Store only the number part

    # Loop through the rest of the rows and pick the second highest, respecting the conditions
    for _, row in df.iloc[1:].iterrows():
        label = row['Label']
        value = row['Value']
        number_part = label[:-1]  # Extract the number part

        # Check if the number part has already been picked
        if number_part in picked_numbers:
            continue

        # Check if the label shares an edge or corner with any previously picked label (same number)
        can_pick = True
        for picked in result:
            if is_adjacent(label, picked['Label']):
                can_pick = False
                break

        # If it passes both checks, add it to the result
        if can_pick:
            result.append(row)
            picked_numbers.add(number_part)

        # Stop if n labels have been picked
        if n and len(result) >= n:
            break

    # Extract just the labels from the result
    final_labels = [r['Label'] for r in result]

    return final_labels
