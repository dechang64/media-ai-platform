#!/usr/bin/env python3
"""
KnowledgeHub - Domain Knowledge Base for Cell Culture Media
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum


class KnowledgeCategory(Enum):
    """Knowledge categories"""
    FORMULATION = "formulation"
    BIOLOGY = "biology"
    OPTIMIZATION = "optimization"
    FL = "fl"


@dataclass
class KnowledgeEntry:
    """Single knowledge entry"""
    id: str
    category: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)


@dataclass
class ComponentInteraction:
    """Media component interaction"""
    component_a: str
    component_b: str
    interaction_type: str  # synergy, antagonism, neutral
    effect: str
    evidence: str


class KnowledgeHub:
    """
    Domain knowledge base for cell culture media.

    Provides curated cell biology literature,
    media component interactions, and optimization guidelines.
    """

    def __init__(self):
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.interactions: Dict[str, List[ComponentInteraction]] = {}
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> None:
        """Initialize with curated knowledge"""

        # ===== FORMULATION KNOWLEDGE =====
        self.add_entry(KnowledgeEntry(
            id="form_001",
            category="formulation",
            title="基础培养基组分",
            content="""
常见基础培养基包括DMEM、RPMI-1640、William's Medium E等。
    每个培养基都有特定的营养成分配置，需要根据细胞类型选择。
            """,
            tags=["基础培养基", "DMEM", "RPMI"],
            citations=["ATCC guidelines"]
        ))

        self.add_entry(KnowledgeEntry(
            id="form_002",
            category="formulation",
            title="血清浓度优化",
            content="""
血清浓度通常在2-20%之间。
    更高浓度促进生长但可能影响分化。
    无血清培养基适用于特定应用。
            """,
            tags=["血清", "无血清", "优化"],
            citations["文献1"]
        ))

        # ===== BIOLOGY KNOWLEDGE =====
        self.add_entry(KnowledgeEntry(
            id="bio_001",
            category="biology",
            title="类器官培养特性",
            content="""
类器官需要特定基质（如Matrigel）来维持3D结构。
    不同来源的类器官有不同培养要求。
            """,
            tags=["类器官", "3D培养", "基质"],
            citations=["Sato et al., 2009"]
        ))

        self.add_entry(KnowledgeEntry(
            id="bio_002",
            category="biology",
            title="细胞代谢需求",
            content="""
癌细胞和正常细胞有不同的代谢需求。
    糖酵解和氧化磷酸化需要平衡。
    乳酸积累会抑制细胞生长。
            """,
            tags=["代谢", "Warburg", "乳酸"],
            citations=["Warburg, 1956"]
        ))

        # ===== OPTIMIZATION KNOWLEDGE =====
        self.add_entry(KnowledgeEntry(
            id="opt_001",
            category="optimization",
            title="响应面方法",
            content="""
响应面方法（RSM）用于优化多组分配方。
   (Box-Behnken设计, Central Composite设计)
            """,
            tags=["RSM", "实验设计", "优化"],
            citations["Box, 1951"]
        ))

        self.add_entry(KnowledgeEntry(
            id="opt_002",
            category="optimization",
            title="贝叶斯优化",
            content="""
贝叶斯优化适合昂贵黑箱函数的优化。
    高斯过程作为代理模型，EI作为采集函数。
            """,
            tags=["贝叶斯", "高斯过程", "EI"],
            citations["Mockus, 1974"]
        ))

        # ===== FEDERATED LEARNING KNOWLEDGE =====
        self.add_entry(KnowledgeEntry(
            id="fl_001",
            category="fl",
            title="联邦学习隐私",
            content="""
联邦学习通过本地训练保护数据隐私。
    差分隐私可以进一步增强保护。
    安全聚合防止模型逆向攻击。
            """,
            tags=["联邦学习", "差分隐私", "安全聚合"],
            citations["McMahan, 2017"]
        ))

        # ===== COMPONENT INTERACTIONS =====
        self.add_interaction(ComponentInteraction(
            component_a="葡萄糖",
            component_b="谷氨酰胺",
            interaction_type="synergy",
            effect="协同提供能量",
            evidence="高葡萄糖+谷氨酰胺促进细胞增殖"
        ))

        self.add_interaction(ComponentInteraction(
            component_a="血清",
            component_b="抗生素",
            interaction_type="antagonism",
            effect="血清降低抗生素效果",
            evidence="血清蛋白结合抗生素"
        ))

    def add_entry(self, entry: KnowledgeEntry) -> None:
        """Add knowledge entry"""
        self.entries[entry.id] = entry

    def add_interaction(self, interaction: ComponentInteraction) -> None:
        """Add component interaction"""
        key = f"{interaction.component_a}_{interaction.component_b}"
        if key not in self.interactions:
            self.interactions[key] = []
        self.interactions[key].append(interaction)

    def search(self, query: str, category: Optional[str] = None,
              limit: int = 10) -> List[KnowledgeEntry]:
        """Search knowledge base"""
        results = []
        query_lower = query.lower()

        for entry in self.entries.values():
            if category and entry.category != category:
                continue

            # Simple keyword matching
            text = (entry.title + " " + entry.content + " " +
                   " ".join(entry.tags)).lower()

            if query_lower in text:
                results.append(entry)

        return results[:limit]

    def get_interactions(self, component: str) -> List[ComponentInteraction]:
        """Get interactions for component"""
        results = []
        for interactions in self.interactions.values():
            for inter in interactions:
                if (inter.component_a == component or
                    inter.component_b == component):
                    results.append(inter)

        return results

    def recommend_formulation(self, cell_type: str,
                           target_application: str) -> Dict[str, Any]:
        """Recommend media formulation based on knowledge"""
        # Search for relevant knowledge
        bio_entries = self.search(cell_type, "biology")
        opt_entries = self.search(target_application, "optimization")

        recommendations = {
            "base_medium": self._infer_base_medium(cell_type),
            "serum_concentration": self._infer_serum(cell_type, target_application),
            "additives": self._infer_additives(cell_type, target_application),
            "knowledge_sources": [e.id for e in bio_entries + opt_entries]
        }

        return recommendations

    def _infer_base_medium(self, cell_type: str) -> str:
        """Infer appropriate base medium"""
        search_results = self.search(cell_type, "biology", limit=3)

        # Simple rules based on cell type
        if any(term in cell_type.lower() for term in ["stem", "类器官"]):
            return "DMEM/F12"
        elif any(term in cell_type.lower() for term in ["immune", "t细胞"]):
            return "RPMI-1640"
        else:
            return "DMEM"

    def _infer_serum(self, cell_type: str, application: str) -> float:
        """Infer serum concentration"""
        if "diff" in application.lower():
            return 2.0  # Lower for differentiation
        elif "primary" in cell_type.lower():
            return 10.0  # Higher for primary cells
        else:
            return 5.0

    def _infer_additives(self, cell_type: str, application: str) -> List[str]:
        """Infer additional supplements"""
        additives = []

        if "类器官" in cell_type:
            additives.extend(["HEPES", "N-acetylcysteine"])

        if "cancer" in cell_type.lower():
            additives.extend(["sodium pyruvate", "glutamine"])

        return additives

    def export_json(self, filepath: str) -> None:
        """Export knowledge base to JSON"""
        data = {
            "entries": [e.__dict__ for e in self.entries.values()],
            "interactions": {k: [i.__dict__ for i in v]
                          for k, v in self.interactions.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_json(self, filepath: str) -> None:
        """Import knowledge base from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for entry_data in data.get("entries", []):
            entry = KnowledgeEntry(**entry_data)
            self.entries[entry.id] = entry


# Global instance
_knowledge_hub = None


def get_knowledge_hub() -> KnowledgeHub:
    """Get global knowledge hub instance"""
    global _knowledge_hub
    if _knowledge_hub is None:
        _knowledge_hub = KnowledgeHub()
    return _knowledge_hub


# Demo usage
if __name__ == "__main__":
    hub = KnowledgeHub()

    # Search
    results = hub.search("类器官")
    print(f"Found {len(results)} results for '类器官'")

    for r in results[:3]:
        print(f"- {r.title}: {r.content[:50]}...")

    # Recommend
    rec = hub.recommend_formulation("肿瘤类器官", "分化")
    print(f"\nRecommendation: {rec}")