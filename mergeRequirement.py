import pkg_resources
from packaging.version import Version, InvalidVersion

def merge_requirements(file1, file2, output_file):
    # Helper function to parse package and version
    def parse_requirements(req_lines):
        req_dict = {}
        for line in req_lines:
            if line.strip() and not line.startswith("#"):
                try:
                    req = pkg_resources.Requirement.parse(line.strip())
                    package_name = req.key
                    version_spec = str(req.specifier) if req.specifier else None
                    
                    # If the package already exists, compare versions
                    if package_name in req_dict:
                        existing_version_spec = req_dict[package_name]
                        existing_version = get_version_from_spec(existing_version_spec)
                        new_version = get_version_from_spec(version_spec)
                        
                        if new_version and (existing_version is None or Version(new_version) > Version(existing_version)):
                            req_dict[package_name] = line.strip()  # Keep the newer version
                    else:
                        req_dict[package_name] = line.strip()  # Add new package if not already present
                except (ValueError, InvalidVersion):
                    pass  # Handle any parsing errors gracefully, skip invalid lines
        return req_dict
    
    # Helper function to extract version from specifier
    def get_version_from_spec(specifier):
        try:
            return str(specifier).split("==")[1] if "==" in specifier else None
        except IndexError:
            return None

    # Read both requirements files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reqs1 = f1.readlines()
        reqs2 = f2.readlines()

    # Parse both requirements files into dictionaries
    req_dict1 = parse_requirements(reqs1)
    req_dict2 = parse_requirements(reqs2)

    # Merge requirements, keeping the latest version
    merged_requirements = {**req_dict1, **req_dict2}

    # Sort the requirements alphabetically (optional)
    sorted_requirements = sorted(merged_requirements.values())

    # Write the merged and sorted requirements to the output file
    with open(output_file, 'w') as outfile:
        outfile.writelines(f"{req}\n" for req in sorted_requirements)

    print(f"Merged requirements saved to {output_file}")

# Usage example
merge_requirements('req/req1.txt', 'req/req2.txt', 'merged_requirements.txt')
