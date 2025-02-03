import requests
import re

def decode_secret_message(doc_url):
    """
    Decodes a secret message from a Google Doc containing character coordinates.
    
    Args:
        doc_url (str): URL of the published Google Doc containing character positions
    """
    # Fetch the document content
    response = requests.get(doc_url)
    if response.status_code != 200:
        print("Failed to fetch document")
        return

    # Extract character positions using regex
    # Looking for patterns like "The character F is in row 0, column 0"
    pattern = r"The character ([A-Z ]) is in row (\d+), column (\d+)"
    matches = re.finditer(pattern, response.text)
    
    # Store character positions and find grid dimensions
    chars = {}
    max_x = max_y = 0
    
    for match in matches:
        char = match.group(1)
        y = int(match.group(2))  # Note: row is y-coordinate
        x = int(match.group(3))  # Note: column is x-coordinate
        chars[(x, y)] = char
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    # Create grid with spaces as default
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    
    # Fill in characters at their specified positions
    for (x, y), char in chars.items():
        grid[y][x] = char
    
    # Print the grid
    print("Decoded message:")
    for row in grid:
        print(''.join(row))

# Use the provided URL
url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
decode_secret_message(url)