#!/bin/zsh
# Fetch the prebuilt sdcpp.xcframework from the latest sdcpp-* GitHub release
# instead of building it from the vendored sources (~10-20 min). Run from
# anywhere; drops the framework into <repo>/Frameworks/.
#
#   ./Scripts/fetch-xcframework.sh
#
# Rebuild from source instead with ./Scripts/build-xcframework.sh (needed
# whenever Sources/CMirage/vendor changes).
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=$(gh release list --repo haplollc/Mirage --json tagName -q \
      '[.[] | select(.tagName | startswith("sdcpp-"))][0].tagName')
[[ -n "$TAG" ]] || { echo "no sdcpp-* release found on haplollc/Mirage" >&2; exit 1; }
echo "fetching sdcpp.xcframework from release $TAG"

TMP=$(mktemp -d)
gh release download "$TAG" --repo haplollc/Mirage \
   --pattern "sdcpp.xcframework.zip" --dir "$TMP"
rm -rf Frameworks/sdcpp.xcframework
mkdir -p Frameworks
ditto -x -k "$TMP/sdcpp.xcframework.zip" Frameworks/
rm -rf "$TMP"
echo "done: Frameworks/sdcpp.xcframework"
