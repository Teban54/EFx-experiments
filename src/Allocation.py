class Allocation:
    def __init__(self, agent_number, item_number, evaluation_matrix):
        self.agent_number = agent_number
        self.item_number = item_number
        self.item_allocation = dict()
        self.agent_allocation = dict()
        for i in range(agent_number):
            self.agent_allocation[i] = set()
        self.evaluation_matrix = evaluation_matrix


    @staticmethod
    def is_valid(id, limit):
        return (id >= 0 and id <= limit)


    def allocate(self, agent_id, item_id):
        if Allocation.is_valid(agent_id, self.agent_number) and Allocation.is_valid(item_id, self.item_number):
            self.item_allocation[item_id] = agent_id
            items = self.agent_allocation[agent_id]
            items.add(item_id)
            return True
        else:
            return False

    def efx_check(self, agent_1, agent_2):
        eval_1 = 0
        eval_2 = 0
        for item in self.agent_allocation[agent_1]:
            eval_1 += self.evaluation_matrix[agent_1][item]
        for item in self.agent_allocation[agent_2]:
            eval_2 += self.evaluation_matrix[agent_1][item]

        efx = True
        max_d = 0
        for item in self.agent_allocation[agent_2]:
            if eval_1 < eval_2 - self.evaluation_matrix[agent_1][item]:
                efx = False
                max_d = max(eval_2 - self.evaluation_matrix[agent_1][item] - eval_1, max_d)
        return efx, max_d

    def is_EFX(self):
        EFX = True
        max_envy = 0
        max_pair = (0, 0)
        for i in range(self.agent_number):
            for j in range(self.agent_number):
                if i != j:
                    efx, envy = self.efx_check(i, j)
                    EFX &= efx
                    if (envy > max_envy):
                        max_envy = envy
                        max_pair = (i, j)

        return EFX, max_pair

    def __str__(self):
        str_out = ""
        for id in range(self.agent_number):
            str_out += "Customer {id}: ".format(id = id)
            str_out += str(self.agent_allocation[id])
            str_out += "\n"
        return str_out