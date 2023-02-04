from setuptools import setup

package_name = 'ros_nml_transforms'

setup(
    name=package_name,
    version='0.0.4',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='a. whit',
    maintainer_email='nml@whit.contact',
    description='A ROS2 package that provides functionality for continuously applying several simple transforms to message data.',
    license='Mozilla Public License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node = ros_nml_transforms.node:main',
            'attractor_node = ros_nml_transforms.attractor_node:main',
        ],
    },
)
