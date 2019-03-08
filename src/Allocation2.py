class Allocation2:
    def __init__(self, agent_number, item_number, evaluation_matrix=None, utility_dict=None, allocations=None):
        self.agent_number = agent_number
        self.item_number = item_number
        self.item_allocation = dict()
        self.agent_allocation = dict()
        for i in range(agent_number):
            self.agent_allocation[i] = set()
        self.evaluation_matrix = evaluation_matrix  # For additive utilities only
        self.utility_dict = utility_dict  # For generic case

        # Initialize if initial allocation is given
        # (allocations[i] is the player that item i is allocated to)
        if allocations is not None:
            for i in range(item_number):
                self.allocate(allocations[i], i)

    @staticmethod
    def is_valid(id, limit):
        return 0 <= id < limit

    def allocate(self, agent_id, item_id):
        if Allocation2.is_valid(agent_id, self.agent_number) and Allocation2.is_valid(item_id, self.item_number):
            # Check if item_id is already allocated to another agent
            if item_id in self.item_allocation.keys():
                old_agent_id = self.item_allocation[item_id]
                self.agent_allocation[old_agent_id].remove(item_id)

            self.item_allocation[item_id] = agent_id
            items = self.agent_allocation[agent_id]
            items.add(item_id)
            return True
        else:
            return False

    def get_allocation(self):
        """
        Return the current allocation of each item.
        :return: New list with the player that each item is allocated to
        """
        return [self.item_allocation[i] for i in range(self.item_number)]

    def evaluate_utility(self, agent_id, items=None):
        """
        Evaluate the utility of a set of items to an agent.
        :param agent_id: Agent
        :param items: Set of items (default is current allocation)
        :return: Utility value
        """
        if items is None:
            items = self.agent_allocation[agent_id]
        if self.utility_dict is not None:
            return self.utility_dict[agent_id][frozenset(items)]
        return sum([self.evaluation_matrix[agent_id][item] for item in items])

    def efx_check(self, agent_1, agent_2):
        eval_1 = self.evaluate_utility(agent_1)
        efx = True
        max_d = 0
        for item in self.agent_allocation[agent_2]:
            items_2 = set(self.agent_allocation[agent_2])
            items_2.remove(item)
            new_utility = self.evaluate_utility(agent_1, items_2)
            if eval_1 < new_utility:
                efx = False
                max_d = max(new_utility - eval_1, max_d)
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

    def utility_measure(self):
        utility = [self.evaluate_utility(i) for i in range(self.agent_number)]
        utility.sort()  # Sorting does not affect local search, so keep it here
        return utility

    def __lt__(self, other):
        assert(self.agent_number == other.agent_number)
        assert(self.item_number == other.item_number)
        utility1 = self.utility_measure()
        utility2 = other.utility_measure()
        for i, u1 in enumerate(utility1):
            u2 = utility2[i]
            if u1 < u2:
                return True
            elif u1 > u2:
                return False
        return True

    def __lt__(self, other):
        assert(self.agent_number == other.agent_number)
        assert(self.item_number == other.item_number)
        utility1 = self.utility_measure()
        utility2 = other.utility_measure()
        for i, u1 in enumerate(utility1):
            u2 = utility2[i]
            if u1 < u2:
                return True
            elif u1 > u2:
                return False
        return True

    def __str__(self):
        str_out = ""
        for id in range(self.agent_number):
            str_out += "Customer {id}: ".format(id = id)
            str_out += str(self.agent_allocation[id])
            str_out += "\n"
        return str_out

