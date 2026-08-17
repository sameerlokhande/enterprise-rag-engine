import time
import chainlit as cl

from src.ingestion.ingestion_graph import build_ingestion_graph
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.cache import SemanticCache
from src.retrieval.reranker import Sub20msReranker
from src.pipeline.confidence import ConfidenceEvaluator
from src.pipeline.generator import GroundedGenerator
from src.telemetry import get_tracer

tracer = get_tracer()
ingestion_pipeline = build_ingestion_graph()

@cl.on_chat_start
async def setup():
    cl.user_session.set("vector_store", QdrantVectorStore())
    cl.user_session.set("semantic_cache", SemanticCache(similarity_threshold=0.92))
    cl.user_session.set("reranker", Sub20msReranker())
    cl.user_session.set("evaluator", ConfidenceEvaluator(alpha_threshold=0.35))
    cl.user_session.set("generator", GroundedGenerator())
    
    await cl.Message(content="⚡ **Enterprise RAG Engine Online** (8GB Local Sandbox Mode with OpenTelemetry Tracing).").send()

@cl.on_message
async def main(message: cl.Message):
    start_time = time.time()
    
    # Trace File Ingestion
    if message.elements:
        for elem in message.elements:
            if isinstance(elem, cl.File):
                async with cl.Step(name=f"Ingesting: {elem.name}") as step:
                    res = ingestion_pipeline.invoke({
                        "file_path": elem.path, "file_type": "", "raw_text": "",
                        "normalized_doc": None, "chunks": {}, "status": "STARTING", "error": ""
                    })
                    if res["status"] == "SUCCESS":
                        step.output = f"✅ Indexed {len(res['chunks']['children'])} child vectors into Qdrant."
                    else:
                        step.output = f"❌ Ingestion Failed: {res['error']}"
        if not message.content.strip(): return

    user_query = message.content
    vector_store = cl.user_session.get("vector_store")
    cache = cl.user_session.get("semantic_cache")
    reranker = cl.user_session.get("reranker")
    evaluator = cl.user_session.get("evaluator")
    generator = cl.user_session.get("generator")

    # ROOT TRACE SPAN
    with tracer.start_as_current_span("rag.user_request") as root_span:
        root_span.set_attribute("user.query", user_query)

        # Step 1: Semantic Cache Check
        async with cl.Step(name="Semantic Cache Lookup") as step1:
            hit = cache.lookup(user_query)
            if hit:
                step1.output = f"⚡ CACHE HIT! (Similarity: {hit['similarity_score']:.4f})"
                root_span.set_attribute("pipeline.outcome", "CACHE_HIT")
                await cl.Message(content=f"⚡ **[Cached Answer]**\n\n{hit['cached_answer']}\n\n*(Served in < 5ms)*").send()
                return
            step1.output = "Cache Miss. Executing RAG Pipeline."

        # Step 2: Vector Search
        async with cl.Step(name="Qdrant Vector Retrieval") as step2:
            candidates = vector_store.search_child_vectors(user_query, top_k=30)
            step2.output = f"Retrieved {len(candidates)} candidate chunks."

        # Step 3: FlashRank Reranking
        async with cl.Step(name="FlashRank Cross-Encoder Rerank") as step3:
            top_chunks = reranker.rerank(user_query, candidates, top_k=3)
            step3.output = f"Filtered down to Top-{len(top_chunks)} passages."

        # Step 4: Confidence Evaluation Gate
        async with cl.Step(name="Confidence Evaluation") as step4:
            should_proceed, score, reason = evaluator.evaluate(top_chunks)
            step4.output = f"Top Relevance Score: {score:.4f} | Status: {reason}"

            if not should_proceed:
                root_span.set_attribute("pipeline.outcome", "SHORT_CIRCUITED")
                await cl.Message(content=f"🔴 **Fallback Triggered**: Low context confidence (`{score:.3f}`). Question rejected.").send()
                return

        # Step 5: Grounded LLM Generation
        async with cl.Step(name="Grounded Ollama Generation") as step5:
            gen_res = generator.generate_response(user_query, top_chunks)
            cache.store(user_query, gen_res["answer"])
            
            latency = (time.time() - start_time) * 1000
            root_span.set_attribute("pipeline.outcome", "SUCCESS")
            root_span.set_attribute("pipeline.total_latency_ms", round(latency, 2))

            response_text = (
                f"{gen_res['answer']}\n\n---\n"
                f"⏱️ **Latency:** `{latency:.0f}ms` | 🎯 **Confidence:** `{score:.2f}` | 📚 **Citations:** {', '.join([f'`{c}`' for c in gen_res['citations']])}"
            )
            await cl.Message(content=response_text).send()