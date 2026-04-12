项目流程
前轮lb、rb是驱动轮
后轮rw是自由轮
当前训练目标为real_car_rigid_sdf.usd
environment是isaaclab基础环境配置
run是基础调试
envs/real_car_env.py是训练环境骨架
train/train_real_car.py是训练环境连通性测试入口

动作接口约定
当前统一采用两轮PWM控制
左轮关节名为lb
右轮关节名为rb
统一动作格式为[left_pwm, right_pwm]
对应控制封装在car_pwm_control.py


碰撞体杂乱（车身自身免碰撞）（已解决sdf）

软体和刚体无法连接在一起（已解决/重构）

软体在小尺度下乱跳（已解决/放大50倍）

* 软体过软（1 铰链/2杨氏模量剪切模量）
