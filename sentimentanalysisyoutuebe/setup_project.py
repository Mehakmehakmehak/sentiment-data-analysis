import os

# Main project directory (assuming we're running this from project root)
project_dirs = [
    "data/raw",
    "data/processed",
    "data/models",
    "notebooks",
    "src/data",
    "src/models",
    "src/visualization",
    "src/utils",
    "app/backend",
    "app/frontend",
    "config",
    "tests"
]

# Create directories
for directory in project_dirs:
    os.makedirs(directory, exist_ok=True)
    # Create an empty __init__.py file in each src directory to make it a proper Python package
    if directory.startswith('src/'):
        with open(f"{directory}/__init__.py", 'w') as f:
            pass

print("Project directory structure created successfully!")