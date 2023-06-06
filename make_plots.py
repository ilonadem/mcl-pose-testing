import argparse

from utils import *

def plot_coord_time(df, axis, axis_title, title):
    plt.scatter(df['time_int'].tolist(),df[axis].tolist())
    plt.xlabel('time (minutes)')
    plt.ylabel(axis_title)
    plt.title(title)

    return plt

def plot_side_by_side(df, vars_list, axis_title, title):
    fig, ax = plt.subplots(nrows=len(vars_list), ncols=2, figsize=(8, 4))
    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.5)

    for i in range(len(vars_list)):
        ax[i,0].scatter(df['time_int'].tolist(),df[f'{vars_list[i]}_X'].tolist(), color='green')
        ax[i,0].set_xlabel('time (minutes)')
        ax[i,0].set_ylabel(f'{vars_list[i]} x coordinate') 
        ax[i,0].set_title(f'{vars_list[i]} x coordinate over time')

        ax[i,1].scatter(df['time_int'].tolist(),df[f'{vars_list[i]}_Y'].tolist(), color='green')
        ax[i,1].set_xlabel('time (minutes)')
        ax[i,1].set_ylabel(f'{vars_list[i]} y coordinate')
        ax[i,1].set_title(f'{vars_list[i]} y coordinate over time')
    
    if args.save:
        fig.savefig(f'{args.keypoint_folder}/{title}_sidebyside.png')

    return fig, ax

# def time_int_to_mins(df):
#     df['time_int'] = df.apply(lambda row: int(row.time[3:5]) + float(row.time[6:])/(60), axis=1)
    
#     return df

parser = argparse.ArgumentParser()
parser.add_argument('--keypoint_folder', default='keypoint_files/patient_kps/', type=str, help='specify file containing keypoint csvs')
# parser.add_argument('--title', type=str)
parser.add_argument('--show', type=bool, default=False)
parser.add_argument('--start', type=float, default=0)
parser.add_argument('--end', type=float, default=-1)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--vars', nargs='+', default=['LEFT_HIP', 'RIGHT_HIP'])
args = parser.parse_args()

for kp_folder in os.listdir(args.keypoint_folder):
    folder = 'keypoint_files/patient_kps/' + kp_folder + '/'
    print("folder:", folder)

    # create and clean dataframe
    kp_df = create_df_from_hour(folder)
    kp_df = add_custom_time(kp_df)
    # kp_df = time_int_to_mins(kp_df)
    kp_df = clean_keypoints(kp_df)
    kp_df = split_x_y(kp_df)
    kp_df = crop_df_by_time(args.start, args.end, kp_df)
    kp_df = time_correct(kp_df)

    print(kp_df.head())

    # plot a simple coordinate
    # for var in args.vars:
        # plt = plot_coord_time(kp_df, f'{var}_X', f'{var} x coordinate', f'{var} x coordinate over time')
        # plt.show()
        
        
        # plt = plot_coord_time(kp_df, f'{var}_Y', f'{var} y coordinate', f'{var} y coordinate over time')
        # plt.show()
        # if args.save:
        #     plt.savefig(f'{args.keypoint_folder}/{var}_y_coordinate_over_time.png')

    if args.save:
        save_dir = f'keypoint_files/visualizations/{kp_folder}/'
        print("SAVE DIR", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + f'/{args.vars}_over_time.png')
    fig, ax = plot_side_by_side(kp_df, args.vars, args.vars, f'{args.vars} coordinates over time')
    plt.show()
        

    # print(kp_df['time_int'])
    # print(type(kp_df['time_int'].tolist()[0]))