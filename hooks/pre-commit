#!/bin/sh
#
# To enable this hook, execute "git config core.hooksPath hooks"

# Read the current version number from the _version.py file
version=$(perl -nle 'print $& if m{(?<=__version__ = ")[^"]*}' _version.py)

# Increment the version number
new_version=$((version + 1))

# Update the version number in the file
sed -i '' "s/__version__ = .*/__version__ = \"$new_version\"/" _version.py

echo "Version number updated to $new_version"
