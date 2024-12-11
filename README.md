# fractal-ome-zarr-hcs-stitching

[Fractal](https://github.com/fractal-analytics-platform) task(s) for registering and fusing OME-Zarr HCS using [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher).

## Development
Find development instructions for building Fractal tasks here: https://github.com/fractal-analytics-platform/fractal-tasks-template/blob/main/DEVELOPERS_GUIDE.md

## Releases
To make new releases, just create a Github release and create a semantic version tag upon release (e.g. v0.1.0). The CI will add the whl files to the release for easier addition to Fractal server & making sure the package is listed on the Fractal task overview.

Alternatively, you can tag a specific commit in main (e.g. with v0.1.0) and push that tag to Github. The CI will take care of making a Github release.
```
git tag v0.1.0
git push origin v0.1.0
```

