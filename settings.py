task_name = 'test_undisen'  # 给当前任务取一个自己能看懂的名字（只能用英文）
city = 'PHO'  # 切换城市 PHO, NYC, SIN
gpuId = "cuda:0"

LS_strategy = 'DoubleTrans'  # 长短期偏好学习策略 TransLSTM, DoubleTrans  TransLSTM现在表示短期用LSTM,长期用Transformer
CL_strategy = 'Triplet'  # 对比学习策略 BPR, Triplet, NativeNCE, CosineNCE

enable_filatt = True  # 以注意力的形式融合长短期偏好

enable_catpre_embedding = True  # 是否采用类别偏好模块

enable_CL = False  # contrastive learning

lstm_layer_num = 1  # 控制lstm层

tfp_layer_num = 1  # 控制transformer块

enable_random_mask = True  # Wonder：random mask
mask_prop = 0.1  # 默认 0.1

enable_cat_pre_mask = False  # 是否对类别轨迹进行掩码
cat_mask_prop = 0.1  # 默认 0.1

enable_random_mask_short = False  # 是否对短期轨迹进行掩码

# 是否进行冷启动实验，即将训练集随机划分4折，验证本模型，本类别偏好模块，相比于CLSPRec更能缓解数据稀疏性问题
cold_start = False  # 是否将训练集划分4折，验证本模型，本类别偏好模块，相比于CLSPRec更能缓解数据稀疏性问题
sparsity = 0.2  # 0.2=2%,0.4=40% 0.6=60% 0.8=80%的数据

if CL_strategy == 'BPR':
    CL_weight = 1
elif CL_strategy == 'Triplet':
    if city == 'PHO':  # 默认0.5
        CL_weight = 0.5
    elif city == 'NYC':
        CL_weight = 0.4
    elif city == 'SIN':
        CL_weight = 0.5
elif CL_strategy == 'NativeNCE':
    CL_weight = 6
elif CL_strategy == 'CosineNCE':
    CL_weight = 10
else:
    raise Exception('CL_strategy error')
enable_alpha = False  # 是否自动学习长期偏好的权重
enable_curriculum_learning = False  # Wonder: curriculum learning，模型训练过程中不断增大对比损失的权重，直到第 curriculum_num 个 epoch 时达到最大权重值 CL_max_weight
curriculum_num = 10
CL_max_weight = 0.2

enable_drop = False  # 为了公平对比，丢弃短期签到中最后的几条签到记录
enable_cal_epoch_metrics = False  # Wonder：是否按照 epoch 来计算 metrics
net_init = False  # 是否进行网络初始化
lr = 1e-5  # 1e-4 或 1e-5
if city == 'PHO':
    epoch = 40  # 默认40
    embed_size = 100  # 论文参数 PHO 100，NYC 40，SIN 60
    run_times = 5  # 整个模型训练的次数 默认为5 要可视化时为1
    drop_steps = 1
elif city == 'NYC':
    epoch = 25
    embed_size = 40  # 20 40 60 80 100
    run_times = 3  # 整个模型训练的次数 默认为3 要可视化时为1
    drop_steps = 2
elif city == 'SIN':  # 默认60
    epoch = 25
    embed_size = 60  # 20 40 60 80 100
    run_times = 3  # 整个模型训练的次数 默认为3 要可视化时为1
    drop_steps = 2
else:
    raise Exception("City name error!")

output_file_name = f'{task_name}_{gpuId}_{city}_{LS_strategy}_Epoch{epoch}_LR' + '{:.0e}'.format(lr)

if enable_drop:
    output_file_name = output_file_name + f'_Drop{drop_steps}'
else:
    output_file_name = output_file_name + "_NoDrop"

if enable_alpha:
    output_file_name = output_file_name + '_Alpha'

if enable_filatt:
    output_file_name = output_file_name + f'_finall_att'  # 以注意力机制融合长短期偏好

if enable_catpre_embedding:
    output_file_name = output_file_name + f'_catpre'
if lstm_layer_num:
    output_file_name = output_file_name + f'_lstm{lstm_layer_num}'

if net_init:
    output_file_name = output_file_name + f'_net_init'

if enable_random_mask:
    output_file_name = output_file_name + f'_Mask{mask_prop}'

if enable_cat_pre_mask:
    output_file_name = output_file_name + f'_catMask_{cat_mask_prop}'

if enable_random_mask_short:
    output_file_name = output_file_name + f'_shortMask_{cat_mask_prop}'

if enable_CL:
    if enable_curriculum_learning:
        output_file_name += f'_{CL_strategy}_{curriculum_num}_{CL_max_weight}'
    else:
        output_file_name += f'_{CL_strategy}{CL_weight}'
else:
    output_file_name = output_file_name + "_NoCL"

output_file_name = output_file_name + '_Embed' + str(embed_size) + f'_trans{tfp_layer_num}'
