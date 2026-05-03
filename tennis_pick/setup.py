from setuptools import setup
import os
from glob import glob

package_name = 'tennis_pick'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Li Wentian',
    maintainer_email='773958366@qq.com',
    description='网球识别抓取 + SLAM 定位 (ROS2 Humble, Jetson + RoboMaster)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tennis_vision_node = tennis_pick.tennis_vision_node:main',
            'tennis_navigation_node = tennis_pick.tennis_navigation_node:main',
            'tennis_manipulation_node = tennis_pick.tennis_manipulation_node:main',
            'tennis_fsm_node = tennis_pick.tennis_fsm_node:main',
        ],
    },
)
