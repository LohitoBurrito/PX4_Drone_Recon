export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lohitoburrito/uav_app/src/dds_config.xml
export CMAKE_PREFIX_PATH=/home/lohitoburrito/libtorch
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export LD_LIBRARY_PATH=/home/lohitoburrito/libtorch/lib:$LD_LIBRARY_PATH


colcon build
source install/local_setup.bash
ros2 run px4_ros_com reconnaissance 