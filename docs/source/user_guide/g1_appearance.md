# G1 body appearance

The G1 MuJoCo models use symmetric silver-gray body shells and dark joint regions.
Load either scene through its existing entry point:

- `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml`
- `decoupled_wbc/control/robot_model/model_data/g1/scene_43dof.xml`

Each scene includes `g1_29dof_with_hand.xml`. Body-color OBJ and PNG assets reside in
the corresponding `meshes/` directory. References are relative to the model package,
so copying the complete G1 directory also includes its appearance assets.

## Color regions

The torso, shoulders, arm shells, and wrist brackets use silver-gray RGB
`(0.68, 0.69, 0.71)`. Dark regions use RGB `(0.07, 0.075, 0.085)`:

| Mesh | Region in the original mesh coordinates |
| --- | --- |
| Left/right `shoulder_yaw_link` | Upper joint seam: `z > 0.024 m` |
| Left/right `shoulder_yaw_link` | Elbow-side cover: `z < -0.036 m` and `x > 0.026 m` |
| Left/right `elbow_link` | Complete tapered distal ring: `x >= 0.0709 m` |

The left and right parts use the same coordinate rules and identical color images.
These regions follow real G1 references and the reviewed teleoperation images.
The wrist bracket beside the distal ring remains silver-gray. Dex3, cables,
fixtures, head, and leg assets retain their existing definitions.

## Mesh and material layout

`*_color.obj` contains the source STL triangles with continuous UV coordinates.
For the elbow mesh, the horizontal texture coordinate follows local `x`. For the
shoulder-yaw mesh, horizontal and vertical coordinates follow local `z` and `x`.
The PNG stores the silver-gray/dark regions. Material names end in `_factory_color`
and use `specular=0.5`, `shininess=0.5`.

The MJCF retains the original mesh names. Visual and coincident collision geoms
reference the same mesh/material, which gives a consistent appearance when a
renderer displays both groups. The OBJ surface coordinates, body masses, inertias,
joint parameters, contact settings, and actuator settings match the source model.
Renderer lighting remains under the application's normal configuration.

URDF consumers continue to use their URDF asset definitions. The body-color
materials described here apply to the MuJoCo XML scenes and their renderers.
