import numpy as np
from abc import abstractmethod


class Entity:
        """ 实体类 """

        @abstractmethod
        def __init__(self, length: int | tuple[int], mode: str = 'random'): """ 写入基因 """

        @abstractmethod
        def cross(self) -> tuple['Entity']: """ 基因交叉 """

        @abstractmethod
        def mutate(self) -> 'Entity': """ 基因突变 """

        @abstractmethod
        def express(self): """ 基因表达 """

        @abstractmethod
        def __str__(self): """ 观察基因表达结果 """
        
        
def genetic_algorithm(category_cls: Entity, n_entities: int, n_epochs: int):
        """ 遗传算法
        `category`-实体种类
        `n_entities`-实体个数
        `n_epochs`-迭代次数"""
        global best_model

        #初始化种群
        entities = [category_cls(length=10) for i in range(n_entities)]  #生成实体列表

        #开始迭代
        for epoch in range(n_epochs):  #使用trange展示进度
                #个体适应度计算
                #fitness_list = [ett.express() for ett in entities]
                #print(fitness_list)
                #交叉-父代
                children = [p0.crosswith(p1) for p0, p1 in
                            zip(entities[:n_entities // 2], entities[1:n_entities // 2 + 1])]
                #变异-子代
                for child in children:
                        child.mutate(0.1)#基数越大，变异概率可适当调整增加
                #选择-新代
                        #消除不合理个体
                entities_cache = entities
                entities = []
                for ett in entities_cache:#除去某个点数少于5的
                        belong_mat = ett.belong_mat
                        is_out = False
                        for i in range(4):
                                if np.sum(belong_mat==i) < 5:
                                        is_out = True
                        if not is_out:
                                entities.append(ett)
                             
                #执行选择淘汰   
                entities = sorted(entities + children, key=lambda x: x.express())  #路径越少越靠前
                entities = entities[:n_entities] if len(entities) > n_entities else entities  #让总数保留在一定范围内
                
                best_model = entities[0]
                

        #输出考前的5个结果
        for i in range(3):
                print(f'NO.{i + 1}: {entities[i]}')
                
        return entities[0]


