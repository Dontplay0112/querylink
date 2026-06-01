from .baseagent import BaseAgent
from .querylink import QueryLinkAgent
from .querylink4amem import QueryLinkAmemAgent
from .querylinkmotivation import QueryLinkMotivation

agent_registry = {
    "QueryLink": QueryLinkAgent,
    "QueryLinkLocomo": QueryLinkAgent,
    "QueryLinkMotivation": QueryLinkMotivation,
    # for plugin test
    "QueryLinkAmem": QueryLinkAmemAgent,
    "QueryLinkAmemLocomo": QueryLinkAmemAgent,
}

__all__ = ["QueryLinkAgent", "QueryLinkAmemAgent", "QueryLinkMotivation"]
def get_agent(agent_name: str) -> BaseAgent:
    agent = agent_registry.get(agent_name)
    if not agent:
        raise ValueError(f"Unknown agent: {agent_name}")
    return agent