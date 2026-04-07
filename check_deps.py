import pkg_resources

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

missing = []    
with open('deps_log.txt', 'w') as out:
    for req in requirements:
        if not req.strip() or req.startswith('#'): continue
        try:
            pkg_resources.require(req)
            out.write(f"Installed: {req}\n")
        except Exception as e:
            missing.append(req)
            out.write(f"Missing/Error: {req} - {e}\n")

    if missing:
        out.write("\nThe following packages are missing or have errors:\n")
        for m in missing: out.write(" - " + m + "\n")
    else:
        out.write("\nAll dependencies seem to be installed.\n")
