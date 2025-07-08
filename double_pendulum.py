import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DoublePendulum:
    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        """
        コンストラクタ
        L1, L2: 振り子の長さ [m]
        m1, m2: 振り子の質量 [kg]
        g: 重力加速度 [m/s^2]
        """
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g

    def derivatives(self, state, t):
        """
        運動方程式を定義する
        state: [theta1, omega1, theta2, omega2]
        t: time
        returns: [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]
        """
        theta1, omega1, theta2, omega2 = state
        g = self.g
        m1 = self.m1
        m2 = self.m2
        L1 = self.L1
        L2 = self.L2

        delta_theta = theta2 - theta1

        # 第1振り子の角加速度 (domega1_dt)
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta_theta)**2
        num1 = -m2 * L1 * omega1**2 * np.sin(delta_theta) * np.cos(delta_theta) \
               + m2 * g * np.sin(theta2) * np.cos(delta_theta) \
               + m2 * L2 * omega2**2 * np.sin(delta_theta) \
               - (m1 + m2) * g * np.sin(theta1)
        domega1_dt = num1 / den1

        # 第2振り子の角加速度 (domega2_dt)
        den2 = (L2 / L1) * den1 # Specification.md の式に基づく
        num2 = -m2 * L2 * omega2**2 * np.sin(delta_theta) * np.cos(delta_theta) \
               + (m1 + m2) * g * np.sin(theta1) * np.cos(delta_theta) \
               - (m1 + m2) * L1 * omega1**2 * np.sin(delta_theta) \
               - (m1 + m2) * g * np.sin(theta2)
        domega2_dt = num2 / den2

        dtheta1_dt = omega1
        dtheta2_dt = omega2

        return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

    def simulate(self, initial_conditions, t_span, dt=0.01):
        """
        シミュレーションを実行する
        initial_conditions: [theta1_0, omega1_0, theta2_0, omega2_0]
        t_span: (t_start, t_end)
        dt: time step
        returns: solution (times, theta1, omega1, theta2, omega2)
        """
        t_start, t_end = t_span
        times = np.arange(t_start, t_end + dt, dt)

        # odeintの初期状態は (theta1, omega1, theta2, omega2)
        solution = odeint(self.derivatives, initial_conditions, times)
        return times, solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]

    def get_cartesian_coords(self, theta1, theta2):
        """
        極座標をデカルト座標に変換する
        theta1, theta2: 角度の時系列データ
        returns: (x1, y1, x2, y2)
        """
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        return x1, y1, x2, y2

if __name__ == '__main__':
    # 簡単なテストと動作確認（後でmain関数に統合）
    pendulum = DoublePendulum()

    # 初期条件: (theta1, omega1, theta2, omega2)
    # 例: 振り子1を90度、振り子2を0度の位置から静かに離す
    initial_state = [np.pi/2, 0, np.pi/2, 0]
    t_span = (0, 20) # 0秒から20秒まで
    dt = 0.01

    times, theta1_vals, omega1_vals, theta2_vals, omega2_vals = pendulum.simulate(initial_state, t_span, dt)

    x1, y1, x2, y2 = pendulum.get_cartesian_coords(theta1_vals, theta2_vals)

    # 結果の簡単なプロット（動作確認用）
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times, theta1_vals, label='Theta1')
    plt.plot(times, theta2_vals, label='Theta2')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.title('Pendulum Angles')

    plt.subplot(1, 2, 2)
    plt.plot(x1, y1, label='Bob 1 path')
    plt.plot(x2, y2, label='Bob 2 path', alpha=0.7)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.title('Pendulum Paths')
    plt.axis('equal') # アスペクト比を保持

    plt.tight_layout()
    plt.show()

    print("DoublePendulum class implemented and basic simulation test run.")

# --- アニメーション機能 ---

def create_animation(pendulum_obj, initial_conditions, t_span, dt=0.01, filename="double_pendulum_single.gif", save_anim=False):
    """
    単一の二重振り子のアニメーションを作成する
    """
    times, theta1, omega1, theta2, omega2 = pendulum_obj.simulate(initial_conditions, t_span, dt)
    x1, y1, x2, y2 = pendulum_obj.get_cartesian_coords(theta1, theta2)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-(pendulum_obj.L1+pendulum_obj.L2)*1.1, (pendulum_obj.L1+pendulum_obj.L2)*1.1), ylim=(-(pendulum_obj.L1+pendulum_obj.L2)*1.1, (pendulum_obj.L1+pendulum_obj.L2)*1.1))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue') # 振り子
    trace, = ax.plot([], [], '.-', lw=1, ms=2, color='red', alpha=0.5) # 第2振り子の軌跡
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # 初期条件をテキストで表示
    initial_state_text = (f"Initial Conditions:\n"
                          f"$\\theta_1 = {initial_conditions[0]:.2f}$ rad\n"
                          f"$\\omega_1 = {initial_conditions[1]:.2f}$ rad/s\n"
                          f"$\\theta_2 = {initial_conditions[2]:.2f}$ rad\n"
                          f"$\\omega_2 = {initial_conditions[3]:.2f}$ rad/s")
    ax.text(0.05, 0.05, initial_state_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return line, trace, time_text

    def animate(i):
        # 振り子の腕
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)

        # 第2振り子の軌跡 (全フレームではなく、過去数フレームなど調整可能)
        # ここでは全軌跡を表示
        trace.set_data(x2[:i+1], y2[:i+1])

        time_text.set_text(time_template % (times[i]))
        return line, trace, time_text

    ani = animation.FuncAnimation(fig, animate, range(0, len(times)),
                                  interval=dt*1000, blit=True, init_func=init)

    if save_anim:
        try:
            print(f"Saving animation to {filename}...")
            if filename.endswith(".gif"):
                ani.save(filename, writer='imagemagick', fps=1/dt)
            elif filename.endswith(".mp4"):
                writer = animation.FFMpegWriter(fps=1/dt, metadata=dict(artist='Me'), bitrate=1800)
                ani.save(filename, writer=writer)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Attempting to save as HTML...")
            try:
                html_filename = filename.split('.')[0] + ".html"
                with open(html_filename, "w") as f:
                    f.write(ani.to_jshtml())
                print(f"Animation saved as HTML: {html_filename}")
            except Exception as e_html:
                print(f"Error saving as HTML: {e_html}")

    plt.title("Single Double Pendulum Animation")
    plt.show()
    return ani


def create_dual_animation(pendulum_obj1, initial_conditions1,
                          pendulum_obj2, initial_conditions2,
                          t_span, dt=0.01, filename="double_pendulum_dual.gif", save_anim=False):
    """
    2つの初期条件による二重振り子のアニメーションを並べて作成する
    """
    times, theta1_1, _, theta2_1, _ = pendulum_obj1.simulate(initial_conditions1, t_span, dt)
    x1_1, y1_1, x2_1, y2_1 = pendulum_obj1.get_cartesian_coords(theta1_1, theta2_1)

    times, theta1_2, _, theta2_2, _ = pendulum_obj2.simulate(initial_conditions2, t_span, dt) # timesは共通
    x1_2, y1_2, x2_2, y2_2 = pendulum_obj2.get_cartesian_coords(theta1_2, theta2_2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    max_L1 = max(pendulum_obj1.L1, pendulum_obj2.L1)
    max_L2 = max(pendulum_obj1.L2, pendulum_obj2.L2)
    plot_limit = (max_L1 + max_L2) * 1.1

    for ax, title_suffix, init_cond in zip([ax1, ax2], ["1", "2"], [initial_conditions1, initial_conditions2]):
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_aspect('equal')
        ax.grid()
        ax.set_title(f"Pendulum {title_suffix}")
        # 初期条件をテキストで表示
        initial_state_text = (f"Initial Conditions:\n"
                              f"$\\theta_1 = {init_cond[0]:.2f}$ rad\n"
                              f"$\\omega_1 = {init_cond[1]:.2f}$ rad/s\n"
                              f"$\\theta_2 = {init_cond[2]:.2f}$ rad\n"
                              f"$\\omega_2 = {init_cond[3]:.2f}$ rad/s")
        ax.text(0.05, 0.05, initial_state_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    line1, = ax1.plot([], [], 'o-', lw=2, markersize=8, color='blue')
    trace1, = ax1.plot([], [], '.-', lw=1, ms=2, color='red', alpha=0.5)

    line2, = ax2.plot([], [], 'o-', lw=2, markersize=8, color='green')
    trace2, = ax2.plot([], [], '.-', lw=1, ms=2, color='purple', alpha=0.5)

    time_template = 'Time = %.1fs'
    time_text_ax1 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
    time_text_ax2 = ax2.text(0.05, 0.9, '', transform=ax2.transAxes) # ax2にも時間表示

    def init():
        line1.set_data([], [])
        trace1.set_data([], [])
        line2.set_data([], [])
        trace2.set_data([], [])
        time_text_ax1.set_text('')
        time_text_ax2.set_text('')
        return line1, trace1, line2, trace2, time_text_ax1, time_text_ax2

    def animate(i):
        # Pendulum 1
        thisx1 = [0, x1_1[i], x2_1[i]]
        thisy1 = [0, y1_1[i], y2_1[i]]
        line1.set_data(thisx1, thisy1)
        trace1.set_data(x2_1[:i+1], y2_1[:i+1])
        time_text_ax1.set_text(time_template % (times[i]))

        # Pendulum 2
        thisx2 = [0, x1_2[i], x2_2[i]]
        thisy2 = [0, y1_2[i], y2_2[i]]
        line2.set_data(thisx2, thisy2)
        trace2.set_data(x2_2[:i+1], y2_2[:i+1])
        time_text_ax2.set_text(time_template % (times[i])) # ax2にも時間表示

        return line1, trace1, line2, trace2, time_text_ax1, time_text_ax2

    ani = animation.FuncAnimation(fig, animate, frames=range(0, len(times)),
                                  interval=dt*1000*1, blit=True, init_func=init) # interval調整

    if save_anim:
        try:
            print(f"Saving dual animation to {filename}...")
            if filename.endswith(".gif"):
                ani.save(filename, writer='imagemagick', fps=max(1,int(1/dt/2))) # fps調整
            elif filename.endswith(".mp4"):
                writer = animation.FFMpegWriter(fps=max(1,int(1/dt/2)), metadata=dict(artist='Me'), bitrate=1800) # fps調整
                ani.save(filename, writer=writer)
            print(f"Dual animation saved to {filename}")
        except Exception as e:
            print(f"Error saving dual animation: {e}")
            print("Attempting to save as HTML...")
            try:
                html_filename = filename.split('.')[0] + ".html"
                with open(html_filename, "w") as f:
                    f.write(ani.to_jshtml())
                print(f"Dual animation saved as HTML: {html_filename}")
            except Exception as e_html:
                print(f"Error saving dual animation as HTML: {e_html}")

    fig.suptitle("Dual Double Pendulum Animation Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitleとの重なりを避ける
    plt.show()
    return ani


# --- 軌道プロット機能 ---
def plot_trajectories(pendulum_obj1, initial_conditions1,
                      pendulum_obj2, initial_conditions2,
                      t_span, dt=0.01, filename="double_pendulum_trajectories.png", save_plot=False):
    """
    二重振り子の軌道を静止画として可視化する。
    - 並列表示: 2つの初期条件による振り子を左右に配置
    - 軌跡表示: 第2振り子の先端軌跡をXY平面で表示
    - 最終位置: シミュレーション終了時の振り子の位置を表示
    - 英語表記: フォントエラー回避のため、グラフ内の文字は英語で表示
    """
    times, theta1_1, _, theta2_1, _ = pendulum_obj1.simulate(initial_conditions1, t_span, dt)
    x1_1, y1_1, x2_1, y2_1 = pendulum_obj1.get_cartesian_coords(theta1_1, theta2_1)

    times, theta1_2, _, theta2_2, _ = pendulum_obj2.simulate(initial_conditions2, t_span, dt)
    x1_2, y1_2, x2_2, y2_2 = pendulum_obj2.get_cartesian_coords(theta1_2, theta2_2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Double Pendulum Trajectories and Final Positions", fontsize=16)

    max_L1 = max(pendulum_obj1.L1, pendulum_obj2.L1)
    max_L2 = max(pendulum_obj1.L2, pendulum_obj2.L2)
    plot_limit = (max_L1 + max_L2) * 1.1

    # Plot for Pendulum 1
    ax1.plot(x2_1, y2_1, 'r-', alpha=0.6, label='Bob 2 Trajectory') # 第2振り子の軌跡
    ax1.plot([0, x1_1[-1], x2_1[-1]], [0, y1_1[-1], y2_1[-1]], 'o-', lw=2, markersize=8, color='blue', label='Final Position') # 最終位置
    ax1.plot(0,0, 'ko', markersize=10) # Pivot
    ax1.set_xlim(-plot_limit, plot_limit)
    ax1.set_ylim(-plot_limit, plot_limit)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title("Pendulum 1")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    initial_state_text1 = (f"Initial Conditions:\n"
                           f"$\\theta_1 = {initial_conditions1[0]:.2f}$ rad\n"
                           f"$\\omega_1 = {initial_conditions1[1]:.2f}$ rad/s\n"
                           f"$\\theta_2 = {initial_conditions1[2]:.2f}$ rad\n"
                           f"$\\omega_2 = {initial_conditions1[3]:.2f}$ rad/s")
    ax1.text(0.05, 0.05, initial_state_text1, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax1.legend(loc='upper right')

    # Plot for Pendulum 2
    ax2.plot(x2_2, y2_2, 'm-', alpha=0.6, label='Bob 2 Trajectory') # 第2振り子の軌跡
    ax2.plot([0, x1_2[-1], x2_2[-1]], [0, y1_2[-1], y2_2[-1]], 'o-', lw=2, markersize=8, color='green', label='Final Position') # 最終位置
    ax2.plot(0,0, 'ko', markersize=10) # Pivot
    ax2.set_xlim(-plot_limit, plot_limit)
    ax2.set_ylim(-plot_limit, plot_limit)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_title("Pendulum 2")
    ax2.set_xlabel("X Position (m)")
    # ax2.set_ylabel("Y Position (m)") # Yラベルは左側にあれば十分
    initial_state_text2 = (f"Initial Conditions:\n"
                           f"$\\theta_1 = {initial_conditions2[0]:.2f}$ rad\n"
                           f"$\\omega_1 = {initial_conditions2[1]:.2f}$ rad/s\n"
                           f"$\\theta_2 = {initial_conditions2[2]:.2f}$ rad\n"
                           f"$\\omega_2 = {initial_conditions2[3]:.2f}$ rad/s")
    ax2.text(0.05, 0.05, initial_state_text2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax2.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitleとの重なりを避ける

    if save_plot:
        try:
            print(f"Saving plot to {filename}...")
            plt.savefig(filename, dpi=300)
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    plt.show()


# --- メイン処理 ---
def main():
    """
    メイン処理
    1. 二重振り子オブジェクトを生成する
    2. 初期条件1・2を設定する（微小差あり）
    3. シミュレーションを実行する（時間範囲: 0〜20秒、刻み: 0.01秒）
    4. アニメーションを作成・表示する（初期値を図内に記載）
    5. 軌道プロットを作成・表示する（静止画、初期値を図内に記載）
    6. GIF/MP4/静止画を保存する（オプション）
    """

    # 保存設定フラグ
    SAVE_GIF = False
    SAVE_VIDEO = False # MP4
    SAVE_PLOTS = False # PNG

    # 二重振り子オブジェクトのパラメータ
    L1, L2 = 1.0, 1.0  # 振り子の長さ [m]
    m1, m2 = 1.0, 1.0  # 振り子の質量 [kg]
    g = 9.81           # 重力加速度 [m/s^2]

    pendulum1 = DoublePendulum(L1=L1, L2=L2, m1=m1, m2=m2, g=g)
    # 2つ目の振り子も同じパラメータで作成 (異なるパラメータも可能)
    pendulum2 = DoublePendulum(L1=L1, L2=L2, m1=m1, m2=m2, g=g)

    # 初期条件 [theta1, omega1, theta2, omega2]
    # 角度はラジアンで指定
    initial_conditions1 = [np.pi / 2, 0, np.pi / 2, 0]  # 振り子1: 両方90度から静かに離す
    initial_conditions2 = [np.pi / 2, 0, np.pi / 2 + 0.01, 0] # 振り子2: 振り子2の初期角度を少しだけずらす

    # シミュレーションのパラメータ
    t_start = 0      # 開始時刻 [s]
    t_end = 20       # 終了時刻 [s]
    dt = 0.01        # 時間刻み [s]
    t_span = (t_start, t_end)

    print("Starting Double Pendulum Simulation...")

    # 1. 単一アニメーションのテスト (オプション)
    # print("\n--- Single Animation Test ---")
    # create_animation(pendulum1, initial_conditions1, t_span, dt,
    #                  filename="double_pendulum_single.gif", save_anim=SAVE_GIF)

    # 2. デュアルアニメーションの作成と表示
    print("\n--- Dual Animation ---")
    dual_anim_filename_gif = "double_pendulum_dual_comparison.gif"
    dual_anim_filename_mp4 = "double_pendulum_dual_comparison.mp4"

    # GIF保存がTrueならGIFで、そうでなければMP4保存がTrueならMP4で保存
    dual_anim_filename = None
    if SAVE_GIF:
        dual_anim_filename = dual_anim_filename_gif
    elif SAVE_VIDEO: # SAVE_GIFがFalseの場合のみMP4を試行
        dual_anim_filename = dual_anim_filename_mp4

    create_dual_animation(pendulum1, initial_conditions1,
                          pendulum2, initial_conditions2,
                          t_span, dt,
                          filename=dual_anim_filename if (SAVE_GIF or SAVE_VIDEO) else "temp_anim_name", # 保存しない場合は一時的な名前
                          save_anim=(SAVE_GIF or SAVE_VIDEO))

    # 3. 軌道プロットの作成と表示
    print("\n--- Trajectory Plot ---")
    plot_filename = "double_pendulum_trajectories.png"
    plot_trajectories(pendulum1, initial_conditions1,
                      pendulum2, initial_conditions2,
                      t_span, dt,
                      filename=plot_filename, save_plot=SAVE_PLOTS)

    print("\nSimulation and visualization complete.")
    if SAVE_GIF or SAVE_VIDEO or SAVE_PLOTS:
        print("Check the output files if saving was enabled.")

if __name__ == '__main__':
    # # 簡単なテストと動作確認（main関数に統合前のもの）
    # pendulum = DoublePendulum()
    # initial_state = [np.pi/2, 0, np.pi/2, 0]
    # t_span = (0, 20)
    # dt = 0.01
    # times, theta1_vals, omega1_vals, theta2_vals, omega2_vals = pendulum.simulate(initial_state, t_span, dt)
    # x1, y1, x2, y2 = pendulum.get_cartesian_coords(theta1_vals, theta2_vals)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(times, theta1_vals, label='Theta1')
    # plt.plot(times, theta2_vals, label='Theta2')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (rad)')
    # plt.legend()
    # plt.title('Pendulum Angles')
    # plt.subplot(1, 2, 2)
    # plt.plot(x1, y1, label='Bob 1 path')
    # plt.plot(x2, y2, label='Bob 2 path', alpha=0.7)
    # plt.xlabel('X position (m)')
    # plt.ylabel('Y position (m)')
    # plt.legend()
    # plt.title('Pendulum Paths')
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()
    # print("DoublePendulum class implemented and basic simulation test run.")

    main()
