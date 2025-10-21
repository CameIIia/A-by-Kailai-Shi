#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A*算法演示程序
使用八数码问题展示A*算法的核心概念和搜索过程

==========================================
A*算法流程总览
==========================================

【算法伪代码】:
1. 初始化OPEN表（优先队列）和CLOSED表（已访问集合）
2. 将起始状态加入OPEN表
3. 重复以下步骤直到OPEN表为空或找到目标：
   a) 从OPEN表中取出f(n)值最小的节点current
   b) 将current加入CLOSED表
   c) 如果current是目标状态，返回解路径
   d) 否则，扩展current的所有邻居节点
   e) 对每个邻居节点neighbor：
      - 如果neighbor不在CLOSED表中，将其加入OPEN表
4. 如果OPEN表为空，返回无解

【关键公式】:
- f(n) = g(n) + h(n)
- g(n): 从起始状态到当前状态n的实际代价
- h(n): 从当前状态n到目标状态的启发式估计代价（必须可采纳）
- 可采纳性: h(n) ≤ h*(n)，其中h*(n)是真实最优代价

【八数码问题中的应用】:
- 状态表示: 3×3网格的数字排列
- 操作: 空格与相邻数字交换位置
- g(n): 移动步数
- h(n): 曼哈顿距离（可采纳的启发式函数）

==========================================
"""
import heapq
from typing import List, Tuple, Optional

class EightPuzzleState:
    """
    八数码状态类 - A*算法的核心数据结构
    
    包含A*算法的三个关键组件：
    1. g(n): 实际代价函数
    2. h(n): 启发式函数  
    3. f(n): 评估函数
    """
    def __init__(self, board: Tuple[int, ...], goal: Tuple[int, ...], 
                 moves: int = 0, parent: Optional['EightPuzzleState'] = None):
        # ==========================================
        # 【A*算法核心数据】
        # ==========================================
        self.board = board      # 当前棋盘状态（问题的状态表示）
        self.moves = moves      # g(n): 从起始状态到当前状态的实际代价
        self.parent = parent    # 父节点指针（用于路径回溯）
        self.goal = goal        # 目标状态
        
        # ==========================================
        # 【A*算法评估函数计算】
        # ==========================================
        self.h = self.manhattan_distance()  # h(n): 启发式函数值（可采纳的）
        self.f = self.moves + self.h        # f(n) = g(n) + h(n): A*评估函数
    
    def manhattan_distance(self) -> int:
        """
        【A*启发式函数】计算曼哈顿距离
        
        这是一个可采纳的启发式函数，因为：
        - 每个数字牌到目标位置的曼哈顿距离 ≤ 实际最少移动步数
        - 总和永远不会高估真实代价
        """
        distance = 0
        for i in range(9):
            if self.board[i] == 0:  # 跳过空格
                continue
            
            # 当前位置坐标
            curr_row, curr_col = divmod(i, 3)
            
            # 目标位置坐标
            target_pos = self.goal.index(self.board[i])
            target_row, target_col = divmod(target_pos, 3)
            
            # 曼哈顿距离 = |行差| + |列差|
            distance += abs(curr_row - target_row) + abs(curr_col - target_col)
        
        return distance
    
    def get_neighbors(self) -> List['EightPuzzleState']:
        """
        【A*状态扩展】生成所有可能的后继状态
        
        这是A*算法中的状态转移函数，生成当前状态的所有合法后继状态
        """
        neighbors = []
        empty_pos = self.board.index(0)  # 找到空格位置
        row, col = divmod(empty_pos, 3)
        
        # 四个移动方向：上、下、左、右
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界条件
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_pos = new_row * 3 + new_col
                
                # 生成新状态：交换空格和相邻数字
                new_board = list(self.board)
                new_board[empty_pos], new_board[new_pos] = new_board[new_pos], new_board[empty_pos]
                
                # 创建新状态节点：g(n) = 父节点g(n) + 1
                neighbors.append(EightPuzzleState(tuple(new_board), self.goal, self.moves + 1, self))
        
        return neighbors
    
    def __lt__(self, other):
        """
        【A*优先队列排序】节点比较函数
        
        A*算法的核心：总是选择f(n)值最小的节点进行扩展
        - 主要按f(n)排序
        - f(n)相等时按h(n)排序（更接近目标的优先）
        """
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

def a_star_solve(start: Tuple[int, ...], goal: Tuple[int, ...], show_steps: bool = True) -> Optional[List[EightPuzzleState]]:
    """
    A*算法求解函数
    
    参数:
    - start: 初始状态
    - goal: 目标状态  
    - show_steps: 是否显示搜索步骤
    
    返回:
    - 解路径，如果无解返回None
    """
    
    # ==========================================
    # 【步骤1】算法初始化
    # ==========================================
    start_state = EightPuzzleState(start, goal)
    open_list = [start_state]  # OPEN表：待扩展节点的优先队列（按f值排序）
    closed_set = set()         # CLOSED表：已扩展节点的集合
    step_count = 0
    
    if show_steps:
        print("开始A*搜索...")
        print(f"初始状态的h(n) = {start_state.h}")
        print("=" * 40)
    
    # ==========================================
    # 【步骤2】主搜索循环
    # ==========================================
    while open_list:
        # ==========================================
        # 【步骤2.1】从OPEN表中选择f值最小的节点
        # ==========================================
        current = heapq.heappop(open_list) # heappop(heap): 弹出并返回堆heap中的最小项
        
        # ==========================================
        # 【步骤2.2】检查重复状态（剪枝优化）
        # ==========================================
        if current.board in closed_set:
            continue  # 跳过已访问的状态
        
        # ==========================================
        # 【步骤2.3】将当前节点加入CLOSED表
        # ==========================================
        closed_set.add(current.board)
        step_count += 1
        
        if show_steps:
            print(f"步骤 {step_count-1}: 扩展节点 g={current.moves}, h={current.h}, f={current.f}")
            print_board(current.board)
            print()
        
        # ==========================================
        # 【步骤2.4】目标检测
        # ==========================================
        if current.board == goal:
            if show_steps:
                print("找到解！")
                print(f"总共扩展了 {step_count} 个节点")
            return build_path(current)
        
        # ==========================================
        # 【步骤2.5】节点扩展 - 生成后继状态
        # ==========================================
        for neighbor in current.get_neighbors():
            # 只将未访问的邻居节点加入OPEN表
            if neighbor.board not in closed_set:
                heapq.heappush(open_list, neighbor)
    
    # ==========================================
    # 【步骤3】搜索失败
    # ==========================================
    return None  # OPEN表为空，无解

def build_path(goal_state: EightPuzzleState) -> List[EightPuzzleState]:
    """
    【A*路径重构】从目标状态回溯到初始状态
    
    A*算法找到目标后，需要通过父节点指针回溯完整路径
    这是A*算法的最后一步：解的构造
    """
    path = []
    current = goal_state
    # 沿着父节点指针回溯
    while current:
        path.append(current)
        current = current.parent
    return path[::-1]  # 反转得到从初始状态到目标状态的正确顺序

def print_board(board: Tuple[int, ...]) -> None:
    """打印棋盘"""
    print("+-------+")
    for i in range(0, 9, 3):
        row = [' ' if board[j] == 0 else str(board[j]) for j in range(i, i+3)]
        print(f"| {' '.join(row)} |")
    print("+-------+")

def demo_solution_path(path: List[EightPuzzleState]) -> None:
    """演示完整解路径"""
    print("\n完整解路径:")
    print("=" * 40)
    
    for i, state in enumerate(path):
        print(f"步骤 {i}: g={state.moves}, h={state.h}, f={state.f}")
        print_board(state.board)
        if i < len(path) - 1:
            print("    ⬇️")
    
    print(f"\n解完成！总共 {len(path)-1} 步")

def demo_case(case_num, title, start, goal):
    """演示单个案例"""
    print(f"\n{'='*60}")
    print(f"案例 {case_num}: {title}")
    print(f"{'='*60}")
    
    print(f"\n初始状态:")
    print_board(start)
    
    print(f"\n目标状态:")
    print_board(goal)
    
    print(f"\n启发式函数计算演示:")
    start_state = EightPuzzleState(start, goal)
    print(f"当前状态的曼哈顿距离 h(n) = {start_state.h}")
    
    # 详细计算过程
    print(f"\n曼哈顿距离详细计算:")
    total = 0
    details = []
    for i in range(9):
        if start[i] == 0:
            continue
        curr_row, curr_col = divmod(i, 3)
        target_pos = goal.index(start[i])
        target_row, target_col = divmod(target_pos, 3)
        dist = abs(curr_row - target_row) + abs(curr_col - target_col)
        if dist > 0:
            details.append(f"  数字 {start[i]}: 当前位置({curr_row},{curr_col}) → 目标位置({target_row},{target_col}) = 距离{dist}")
            total += dist
    
    if details:
        for detail in details:
            print(detail)
        print(f"  总曼哈顿距离: {total}")
    else:
        print("  所有数字牌都在正确位置")
    
    # 执行A*搜索
    print(f"\n执行A*搜索:")
    solution = a_star_solve(start, goal, show_steps=True)
    
    if solution:
        demo_solution_path(solution)
        
        print(f"\n算法性能:")
        print(f"  • 最优解步数: {len(solution)-1}")
        if len(solution) == 2:
            print(f"  • 搜索效率: 极高（只需一步即可到达目标）")
        elif len(solution) <= 6:
            print(f"  • 搜索效率: 很高（启发式函数有效指导搜索）")
        else:
            print(f"  • 搜索效率: 良好（复杂问题需要更多步骤）")
        print(f"  • 解的质量: 最优（A*保证找到最短路径）")

def main():
    """主演示程序"""
    
    # 案例1：简单例子（文档中的例子）
    start1 = (1, 2, 3, 8, 6, 4, 7, 0, 5)
    goal1 = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    demo_case(1, "简单示例（1步解决）", start1, goal1)
    
    # 案例2：复杂例子（用户提供的例子）
    start2 = (2, 8, 3, 1, 6, 4, 7, 0, 5)
    goal2 = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    demo_case(2, "复杂示例（多步求解）", start2, goal2)

if __name__ == "__main__":
    main()