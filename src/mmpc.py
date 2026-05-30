from src.stats import g_square_test

class MMPC:
    def __init__(self, data, alpha=0.05):
        self.data = data
        self.nodes = list(data.columns)
        self.alpha = alpha

    def get_skeleton(self):
        skeleton = set()
        for node in self.nodes:
            neighbors = self._learn_pc(node)
            for neighbor in neighbors:
                skeleton.add(tuple(sorted((node, neighbor))))
        return skeleton
    
    def _learn_pc(self, target):
        pc = []
        remaining = [n for n in self.nodes if n != target]

        while True:
            best_candidate = None
            max_min_p = -1.0

            for x in remaining:
                p_val, _ = g_square_test(self.data, x, target, pc, self.alpha)

                assoc = 1.0 - p_val
                if assoc > max_min_p:
                    max_min_p = assoc
                    best_candidate = x
            
            if max_min_p > (1.0 - self.alpha):
                pc.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        for x in pc[:]:
            temp_pc = [n for n in pc if n != x]
            _, is_indep = g_square_test(self.data, x, target, temp_pc, self.alpha)
            if is_indep:
                pc.remove(x)
        
        return pc
    