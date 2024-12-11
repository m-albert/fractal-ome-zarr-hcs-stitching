# fractal-ome-zarr-hcs-stitching

[Fractal](https://github.com/fractal-analytics-platform) task(s) for registering and fusing OME-Zarr HCS using [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher).

## Development
Find development instructions for building Fractal tasks here: https://github.com/fractal-analytics-platform/fractal-tasks-template/blob/main/DEVELOPERS_GUIDE.md

## Releases
To make new releases, tag a specific commit in main (e.g. with v0.1.0) and push that tag to Github. The CI will take care of making a Github release.
```
git tag v0.1.0
git push origin v0.1.0
```

Alternatively, create a new release with a correspondig tag directly from Github (to be verified that it's working).
