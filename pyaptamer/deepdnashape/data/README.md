# deepDNAshape scaling parameters

`params.json` holds per-feature scaling values (`min`, `max`, `mean`, `std`,
`median`, `percentile_range`, and `method`) used to convert raw model outputs
back to physical DNA-shape units.

These values come from the original deepDNAshape repository:

https://github.com/JinsenLi/deepDNAshape

They are required at inference time by `deepDNAshape` and are shipped with the
package under this `data/` folder.
