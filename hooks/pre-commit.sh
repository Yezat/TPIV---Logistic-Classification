#!/bin/sh
#
# An example hook script to verify what is about to be committed.
# Called by "git commit" with no arguments.  The hook should
# exit with non-zero status after issuing an appropriate message if
# it wants to stop the commit.
#
# To enable this hook, rename this file to "pre-commit".
# Read the current version number from the _version.py file
# Read the current version number
version=$(perl -nle 'print $& if m{(?<=__version__ = ")[^"]*}' _version.py)

# Increment the version number
new_version=$((version + 1))

# Update the version number in the file
sed -i '' "s/__version__ = .*/__version__ = \"$new_version\"/" _version.py

echo "Version number updated to $new_version"
