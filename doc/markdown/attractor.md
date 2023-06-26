<!-- License

Copyright 2022 Neuromechatronics Lab, Carnegie Mellon University (a.whit)

Contributors:
  a. whit. (nml@whit.contact)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
-->

## Attractor (robot)

This package implements a [force field] that functions like an [attractor], to 
pull a robotic end effector toward some point, line, or plane in 3D space. 
The force field is implemented in the file 
[attractor_node.py](ros_nml_transforms/attractor_node.py). Specifically, the 
[`compute_effector_force`] function implements a [linear attractor] of the 
form

        """
        "modeled as a spring+damper_ system that pull the device towards its 
         projection on the constraint segment"
        K_ is the spring constant, or stiffness_.
        damper_
        C is the damping coefficient
        damping_
        
        .. _K: https://en.wikipedia.org/wiki/Spring_(device)#Physics
        .. _stiffness: https://en.wikipedia.org/wiki/Stiffness
        .. _spring+damper: https://en.wikipedia.org/wiki/Mass-spring-damper_model
        .. _damper: https://en.wikipedia.org/wiki/Dashpot
        .. _damping: https://en.wikipedia.org/wiki/Damping
        
### Parameters

Several [ROS2 parameters] are used to configure the attractor.

#### `attractor_basis`

This is a matrix that determines the line or plane (passing through the origin) 
that the robot will be constrained to. It is a subspace basis relative to the 
robot coordinate system. For example, the following parameter value constrains 
the robot to up-down motion: 

```
/robot/transforms:
  ros__parameters:
    effector:
      attractor_basis:
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 1.00
```

This configuration encodes
Move the one to a different diagonal element to constrain to side-side or 
forward-backward motion. Setting two diagonal elements constrains motion to a 
2D plane, and setting a full rank basis should effectively remove all 
constraints.

#### `attractor_stiffness` and `attractor_damping`

The stiffness and damping parameters determine the forces applied to the 
robotic end effector. These should be tuned to each specific context, to avoid 
instabilities and unexpected behavior.

Caution and advance testing are recommended. It is best to be prepared to
press the "cancel" button

[damping ratio]: https://en.wikipedia.org/wiki/Damping#Damping_ratio_definition

### Observations 

* Occasional instabilities have been noticed during development.
* The Novint Falcon does not seem to apply constraints in a directionally 
  homogeneous manner (e.g., it is better at constraining left-right deviation 
  than up-down).
* Even though the Force Dimension example disabled velocity thresholding, but a 
  decision was made to leave it enabled, in this context. that enabled.


### Sample configuration

The following is a sample configuration for the robot with attractor dynamics: 

```yaml
/robot/manipulandum:
  ros__parameters:
    disable_hardware: false
    sample_interval_s: 0.0005
    gravity_compensation: true
    effector_mass_kg: 0.190
/robot/transforms:
  ros__parameters:
    cursor:
      position_transform:
      - +0.00
      - +26.0
      - +0.00
      - +0.00
      - +0.00
      - +26.0
      - -26.0
      - +0.00
      - +0.00
      force_transform:
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 0.00
    effector:
      attractor_basis:
      - 0.00
      - 0.00
      - 0.00
      - 0.00
      - 1.00
      - 0.00
      - 0.00
      - 0.00
      - 1.00
      attractor_offset:
      - 0.00
      - 0.00
      - 0.00
      attractor_stiffness: 500.0
      attractor_damping: 20.0
```

<!--
```math
\newcommand\myexp[1]{e^{#1}}
```

$\myexp{-1}$

```math
\def{\mat}{\mathbf}

\mat{P}_d = \vec{p}_d \left( \vec{p}_{d}^{T} \vec{p}_{d} \right)^{-1} 
            \vec{p}_{d}^{T}

\vec{f}_g = \mat{P}_d \cdot K * (\vec{p}_a - \vec{p}) - C * \vec{v}
```
-->
<!--
\renewcommand\vec{\mathbf}

\renewcommand{\vec}[1]{\mathbf{#1}}
-->

<!--
Subscripts indicating the time index have been omitted, but this equation is 
updated at each sample update.
-->

<!--
![System diagram](../../assets/images/system_diagram.svg "System diagram")
-->


<!---------------------------------------------------------------------
   References
---------------------------------------------------------------------->

[attractor]: https://en.wikipedia.org/wiki/Attractor

[force field]: https://en.wikipedia.org/wiki/Force_field_(physics)

[`compute_effector_force`]: https://github.com/ricmua/ros_nml_transforms/blob/c211c19db66085b9754429231457cb978cd66e89/ros_nml_transforms/attractor_node.py#L76

[linear attractor]: https://en.wikipedia.org/wiki/Attractor#Linear_equation_or_system

