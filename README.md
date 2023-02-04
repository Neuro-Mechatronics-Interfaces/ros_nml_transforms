---
title: README
author: a.whit ([email](mailto:nml@whit.contact))
date: February 2023
---

<!-- License

Copyright 2022-2023 Neuromechatronics Lab, Carnegie Mellon University (a.whit)

Created by: a. whit. (nml@whit.contact)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
-->

# ROS2 NML Transforms

This package defines a [ROS2 node] that continuously performs simple 
transformations of message data. This mostly consists of static linear 
transformations applied to numerical data contained in messages.

Transforms are particularly relevant to the Force Dimension robot. For 
example, one transform converts the current robot effector position into a 
position in the coordinate system of a task / GUI space.

## Installation

This package can be added to any [ROS2 workspace]. ROS2 workspaces are built 
using [colcon].

<!--
### Testing

The [doctest]s in this document can be used to quickly verify a successful 
installation. All tests should be invoked from within a [configured ROS2 environment] 
after [installation](#installation) of the workspace. Be sure to 
[source the workspace overlay], in addition to the 
[ROS2 environment][source the ROS2 environment].


```bash
python -m doctest path/to/ros_nml_transforms/README.md
```
-->

<!--
For more formalized unit testing, use the provided [pytest]s.

```bash
python -m pytest path/to/ros_nml_transforms/test/test_ros_nml_transforms.py
```
-->

<!--
## Example

Perhaps the best way to introduce the package and task is via an example.
-->



## License

Copyright 2022-2023 [Neuromechatronics Lab], Carnegie Mellon University

Contributors: 
* a. whit. (nml@whit.contact)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.


<!---------------------------------------------------------------------
   References
---------------------------------------------------------------------->

[Python path]: https://docs.python.org/3/tutorial/modules.html#the-module-search-path

[doctest]: https://docs.python.org/3/library/doctest.html

[pytest]: https://docs.pytest.org/

[ROS2]: https://docs.ros.org/en/humble/index.html

[setuptools]: https://setuptools.pypa.io/en/latest/userguide/quickstart.html#basic-use

[Neuromechatronics Lab]: https://www.meche.engineering.cmu.edu/faculty/neuromechatronics-lab.html

[pip install]: https://pip.pypa.io/en/stable/cli/pip_install/

[ROS2 workspace]: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html

[colcon]: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html

[ROS2 package]: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html#what-is-a-ros-2-package

[ROS2 graph]: https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Nodes/Understanding-ROS2-Nodes.html#background

[ROS2 node]: https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Nodes/Understanding-ROS2-Nodes.html#background

[configured ROS2 environment]: https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html

[source the workspace overlay]: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html#source-the-overlay

[source the ROS2 environment]: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html#source-ros-2-environment


