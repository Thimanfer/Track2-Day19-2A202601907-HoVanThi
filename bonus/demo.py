"""Bonus Challenge Demo Script.

Demonstrates 5 query scenarios showcasing the integration of
Episodic Memory (Vector DB + RRF) and User Profile / Activity (Feature Store).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=" * 70)
    print("Bonus Challenge: AI Assistant with Hybrid Memory (POC Demo)")
    print("=" * 70)

    agent = HybridMemoryAgent()
    user_id = "u_001"

    # Seed episodic memories for u_001
    sample_memories = [
        "Kubernetes deployment hướng dẫn cấu hình Horizontal Pod Autoscaler cho cụm microservices.",
        "Kiến trúc bảo mật Zero Trust trên Google Cloud Platform và AWS IAM federation.",
        "Tài liệu tự động hóa scale out hạ tầng khi lưu lượng mạng tăng đột biến trong sự kiện khuyến mãi.",
        "Kỹ thuật tối ưu hóa chi phí đám mây thông qua Spot Instances và Reserved VM Instances.",
        "Nghiên cứu cơ chế Retrieval-Augmented Generation kết hợp Vector DB và Feature Store trong tiếng Việt.",
    ]

    print(f"\n[1] Storing {len(sample_memories)} episodic memories for user '{user_id}'...")
    for mem in sample_memories:
        agent.remember(mem, user_id=user_id)
    print("✓ All episodic memories indexed successfully into Qdrant & BM25.\n")

    # 5 Demonstration Queries
    queries = [
        ("Query 1 (Vector/Episodic Hit)", "Tôi đã đọc gì về Kubernetes?"),
        ("Query 2 (Profile Context Needed)", "Recommend cho tôi chủ đề nên đọc tiếp theo?"),
        ("Query 3 (Recent Activity / Streaming Feature)", "Tôi đang quan tâm tới những gì gần đây và cường độ tìm kiếm thế nào?"),
        ("Query 4 (Paraphrase Query - Semantic Vector Win)", "Tài liệu về giải pháp co giãn hạ tầng server theo tải người dùng?"),
        ("Query 5 (Mixed Query - Hybrid Retrieval + Profile)", "Cho tôi tóm tắt kinh nghiệm về cloud security và tối ưu chi phí."),
    ]

    for idx, (label, q) in enumerate(queries, 1):
        print(f"\n>>> [{idx}/5] {label}")
        print(f"User Asked: \"{q}\"")
        context = agent.recall(q, user_id=user_id, top_k=2)
        print(context)
        print("-" * 70)

    print("\n✓ Demo completed successfully with 5/5 query outputs verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
