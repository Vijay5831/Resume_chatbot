def check_import(package_name):
    """Try to import a package and report if it's available."""
    try:
        __import__(package_name)
        return True
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unknown error: {str(e)}"

# List of packages to check
packages = [
    'flask', 
    'dotenv', 
    'PyPDF2', 
    'langchain', 
    'langchain_core', 
    'langchain_google_genai', 
    'langchain_community',
    'langchain_huggingface',
    'faiss',
    'sentence_transformers', 
    'torch'
]

print("Dependency Check Results:")
print("-" * 50)

for package in packages:
    result = check_import(package)
    if result is True:
        print(f"✓ {package}: Installed")
    else:
        error_message = result[1]
        print(f"✗ {package}: {error_message}")

print("-" * 50) 