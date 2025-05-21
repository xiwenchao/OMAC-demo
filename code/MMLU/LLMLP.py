import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer
from prompt_lib import *
from prompt_iteration.chat_service import ChatService
import uuid
from LLM_Neuron import parse_ranks


ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1} # TODO: only 0 is implemented

class LLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo",
                 prompt_info=None, optimized_prompts=None):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype

        prompt_role, role_name, opt_prompt = prompt_info
        self.prompt_info = prompt_info

        if prompt_role == "rank":
            self.rank_prompt = opt_prompt + "\n" + RANK_FORMAT
        elif "rank" in optimized_prompts:
            self.rank_prompt = optimized_prompts["rank"][1] + "\n" + RANK_FORMAT
        else:
            self.rank_prompt = RANK_DEFAULT

        self.optimized_prompts = optimized_prompts
        
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []

        agent_structure = {}
        for idx, agent in enumerate(agent_roles):
            if self.prompt_info[0] == 'structure' or 'structure' in self.optimized_prompts:
                structure_prompt = self.prompt_info[2] if self.prompt_info[0] == 'structure' else self.optimized_prompts['structure'][1]
                agent_structure[agent] = self.get_structure(agent, agent_roles, structure_prompt, self.optimized_prompts)
            else:
                agent_structure[agent] = agent_roles

        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype, self.prompt_info, self.optimized_prompts))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype, self.prompt_info, self.optimized_prompts))
                # print(len(agents_last_round)) !!!
                for a1 in agents_last_round:
                    if a1.role in agent_structure[agent_roles[idx]]:
                        self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")
    
    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            # print("Consensus answer: {}".format(consensus_answer))
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question):
        def get_completions():
            # get completions
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2
        # question = format_question(question, self.qtype)

        # shuffle the order of agents
        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            # print(0, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        loop_indices = list(range(self.agents, self.agents*2))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            # print(1, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents*2))
        for rid in range(2, self.rounds):
            # TODO: compatible with 1/2 agents
            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
            
                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype, self.rank_prompt)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents, tops))
                resp_cnt += self.activation_cost

            loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    # print(rid, idx)
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        completions = get_completions()
        return most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0], resp_cnt, completions, total_prompt_tokens, total_completion_tokens


    def backward(self, result):
        flag_last = False
        for rid in range(self.rounds-1, -1, -1):
            if not flag_last:
                if len([idx for idx in range(self.agents*rid, self.agents*(rid+1)) if self.nodes[idx].active]) > 0:
                    flag_last = True
                else:
                    continue

                ave_w = 1 / len([idx for idx in range(self.agents*rid, self.agents*(rid+1)) if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result)])
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result):
                        self.nodes[idx].importance = ave_w
                    else:
                        self.nodes[idx].importance = 0
            else:
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        return [node.importance for node in self.nodes]

    def get_structure(self, current_agent, candidate_agents, structure_prompt, optimized_prompts):
        # get the structure of the neural network
        # current_agent: the current agent
        # candidate_agents: the candidate agents
        # return: a list of tuples (agent, weight)
        structure = []
    
        system_pt = construct_structure_message(current_agent, candidate_agents, structure_prompt, optimized_prompts)
        service_2 = ChatService(system_pt)
        # logging.info('pre-rank question: {}'.format(PROMPT_INFO[2]))
        rand_uuid = str(uuid.uuid4())
        question = "{{{}}}: ".format(rand_uuid) + """Diretly output your choices of candidate agents.""" + "\n" + PRE_RANK_FORMAT
        try:
            max_retries = 5
            retry_count = 0
            backoff_time = 1  # Initial backoff time in seconds
            
            while retry_count < max_retries:
                try:
                    completion = service_2.ask(question, "gpt-3.5-turbo").rstrip('\n')
                    break  # Successfully got response, exit the retry loop
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Failed after {max_retries} attempts. Last error: {e}")
                        # Set a default completion or re-raise based on your needs
                    else:
                        print(f"Attempt {retry_count} failed: {e}. Retrying in {backoff_time}s...")
                        import time
                        time.sleep(backoff_time)
                        backoff_time *= 2  # Exponential backoff
        except Exception as e:
            print(f"Unexpected error in retry mechanism: {e}")

        tops = parse_ranks(completion, max_num=len(candidate_agents), random_num=7)
        # print('structure answer: {}'.format(completion))
        # print('tops: {}'.format(tops))
        if len(tops) < 2:
            tops.append(random.choice(list(set(range(len(candidate_agents))) - set(tops))))
        structure = [candidate_agents[i] for i in tops]
        return structure
