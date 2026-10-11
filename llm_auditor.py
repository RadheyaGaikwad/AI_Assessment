"""
Augle AI Assessment - Agentic RAG Data Auditor
File: llm_auditor.py
Description: Robust Gemini-powered Agentic RAG auditor supporting local fallback, 
COCO bounding-box validation, category distribution, training metrics, and tool calling.
"""

import os
import json
import csv
import math
import time
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
from google import genai
from google.genai import types


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

ANNOTATIONS_PATH = (
    PROJECT_DIR / "data" / "annotations" / "annotations.json"
)
METRICS_PATH = PROJECT_DIR / "data" / "metrics.csv"

CACHE_DIR = PROJECT_DIR / "data" / "rag_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_CACHE_PATH = CACHE_DIR / "document_embeddings.npz"

# Changed default model to gemini-1.5-flash to avoid free-tier quota exhaustion
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_RETRIEVAL_RESULTS = 5
MAX_API_ATTEMPTS = 2


# =========================================================
# AUDITOR
# =========================================================

class AgenticRAGAuditor:

    def __init__(self):
        print("\n--- Initializing Agentic RAG Data Auditor ---")

        self.coco_data = self._load_json(ANNOTATIONS_PATH)
        self.metrics_data = self._load_csv(METRICS_PATH)

        self.images_by_id = {
            image["id"]: image
            for image in self.coco_data.get("images", [])
            if "id" in image
        }

        self.categories_by_id = {
            category["id"]: category
            for category in self.coco_data.get("categories", [])
            if "id" in category
        }

        self.embedder = None
        self.documents = []
        self.document_embeddings = None

        self.client = None
        self.chat = None
        self.tool_functions = []

        self._initialize_rag()
        self._initialize_chat()

        print("[OK] Auditor initialization finished.\n")

    # =====================================================
    # LOAD PROJECT FILES
    # =====================================================

    @staticmethod
    def _load_json(path):
        if not path.exists():
            print(f"[!] Annotation file not found: {path}")
            return {
                "images": [],
                "annotations": [],
                "categories": [],
            }

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError("COCO JSON must contain an object.")

            return data

        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[!] Could not load annotations: {exc}")
            return {
                "images": [],
                "annotations": [],
                "categories": [],
            }

    @staticmethod
    def _load_csv(path):
        if not path.exists():
            print(f"[!] Metrics file not found: {path}")
            return []

        try:
            with path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                return list(csv.DictReader(file))

        except OSError as exc:
            print(f"[!] Could not load metrics: {exc}")
            return []

    # =====================================================
    # TOOL 1: BOUNDING-BOX VALIDATION
    # =====================================================

    def audit_bounding_boxes(self):
        images = self.coco_data.get("images", [])
        annotations = self.coco_data.get("annotations", [])
        categories = self.coco_data.get("categories", [])

        malformed = 0
        out_of_bounds = 0
        invalid_dimensions = 0
        non_finite = 0
        missing_image_refs = 0
        missing_category_refs = 0
        valid_boxes = 0

        areas = []
        category_counts = Counter()
        images_with_annotations = set()

        category_ids = set(self.categories_by_id)

        for ann in annotations:
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")
            bbox = ann.get("bbox")

            images_with_annotations.add(image_id)
            category_counts[str(category_id)] += 1

            if image_id not in self.images_by_id:
                missing_image_refs += 1

            if categories and category_id not in category_ids:
                missing_category_refs += 1

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                malformed += 1
                continue

            try:
                x, y, width, height = map(float, bbox)
            except (TypeError, ValueError):
                malformed += 1
                continue

            if not all(
                math.isfinite(v) for v in (x, y, width, height)
            ):
                non_finite += 1
                continue

            if width <= 0 or height <= 0:
                invalid_dimensions += 1
                continue

            image = self.images_by_id.get(image_id)

            if image is None:
                continue

            try:
                image_width = float(image["width"])
                image_height = float(image["height"])
            except (KeyError, TypeError, ValueError):
                malformed += 1
                continue

            if (
                not math.isfinite(image_width)
                or not math.isfinite(image_height)
                or image_width <= 0
                or image_height <= 0
            ):
                malformed += 1
                continue

            if (
                x < 0
                or y < 0
                or x + width > image_width + 1
                or y + height > image_height + 1
            ):
                out_of_bounds += 1
            else:
                valid_boxes += 1

            areas.append(width * height)

        result = {
            "total_images": len(images),
            "total_annotations": len(annotations),
            "images_with_annotations": len(images_with_annotations),
            "categories_defined": len(categories),
            "valid_boxes_with_known_image_dimensions": valid_boxes,
            "malformed_annotations": malformed,
            "out_of_bounds_boxes": out_of_bounds,
            "invalid_box_dimensions": invalid_dimensions,
            "non_finite_boxes": non_finite,
            "missing_image_references": missing_image_refs,
            "missing_category_references": missing_category_refs,
            "category_annotation_counts": dict(category_counts),
            "mean_box_area": float(np.mean(areas)) if areas else None,
            "min_box_area": float(np.min(areas)) if areas else None,
            "max_box_area": float(np.max(areas)) if areas else None,
            "limitation": (
                "Geometric validity does not prove that a box visually "
                "surrounds the correct object."
            ),
        }

        return json.dumps(result, indent=2)

    # =====================================================
    # TOOL 2: CATEGORY DISTRIBUTION
    # =====================================================

    def audit_category_distribution(self):
        annotations = self.coco_data.get("annotations", [])
        categories = self.coco_data.get("categories", [])

        counts = Counter(
            str(ann.get("category_id"))
            for ann in annotations
            if ann.get("category_id") is not None
        )

        total = sum(counts.values())

        names = {
            str(category["id"]): category.get(
                "name", str(category["id"])
            )
            for category in categories
            if "id" in category
        }

        distribution = []

        for category_id, count in counts.most_common():
            distribution.append({
                "category_id": category_id,
                "category_name": names.get(
                    category_id, f"Unknown category {category_id}"
                ),
                "annotation_count": count,
                "percentage": round(100 * count / total, 2)
                if total else 0,
            })

        if len(counts) < 2:
            imbalance_ratio = None
            assessment = (
                "Cannot compare class balance: fewer than two "
                "categories have annotations."
            )
        else:
            values = list(counts.values())
            imbalance_ratio = round(max(values) / min(values), 2)
            assessment = (
                "Potential class imbalance detected."
                if imbalance_ratio >= 3
                else "No severe imbalance detected by the 3:1 heuristic."
            )

        return json.dumps({
            "total_annotations": total,
            "categories_with_annotations": len(counts),
            "category_distribution": distribution,
            "max_to_min_count_ratio": imbalance_ratio,
            "assessment": assessment,
            "limitation": (
                "Counts do not measure unique images, object difficulty, "
                "or visual annotation quality."
            ),
        }, indent=2)

    # =====================================================
    # TOOL 3: TRAINING METRICS
    # =====================================================

    def audit_training_logs(self):
        if not self.metrics_data:
            return json.dumps({
                "error": "No training metrics found.",
                "expected_file": str(METRICS_PATH),
            }, indent=2)

        first_row = self.metrics_data[0]
        required = {"train_loss", "val_loss"}

        if not required.issubset(first_row.keys()):
            return json.dumps({
                "error": "Required metrics columns are missing.",
                "required": sorted(required),
                "available": list(first_row.keys()),
            }, indent=2)

        rows = []

        try:
            for index, row in enumerate(self.metrics_data, start=1):
                train_loss = float(row["train_loss"])
                val_loss = float(row["val_loss"])

                if not (
                    math.isfinite(train_loss)
                    and math.isfinite(val_loss)
                ):
                    return json.dumps({
                        "error": f"Non-finite loss in row {index}."
                    }, indent=2)

                rows.append({
                    "epoch": int(row.get("epoch") or index),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

        except (TypeError, ValueError) as exc:
            return json.dumps({
                "error": f"Could not parse metrics: {exc}"
            }, indent=2)

        if not rows:
            return json.dumps({"error": "No metric rows."}, indent=2)

        best = min(rows, key=lambda row: row["val_loss"])

        return json.dumps({
            "metrics_file": str(METRICS_PATH),
            "epochs_recorded": len(rows),
            "first_epoch": rows[0],
            "final_epoch": rows[-1],
            "best_validation_epoch": best["epoch"],
            "best_validation_loss": best["val_loss"],
            "training_loss_decreased": (
                rows[-1]["train_loss"] < rows[0]["train_loss"]
            ),
            "validation_loss_decreased": (
                rows[-1]["val_loss"] < rows[0]["val_loss"]
            ),
            "epoch_metrics": rows,
            "limitation": (
                "Loss alone does not establish detection accuracy. "
                "Evaluate predictions with appropriate detection metrics."
            ),
        }, indent=2)

    # =====================================================
    # TOOL 4: SMALLEST BOXES
    # =====================================================

    def find_smallest_boxes(self, limit=10):
        limit = max(1, min(int(limit), 50))
        results = []

        for ann in self.coco_data.get("annotations", []):
            bbox = ann.get("bbox")

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            try:
                x, y, width, height = map(float, bbox)
            except (TypeError, ValueError):
                continue

            if (
                not all(
                    math.isfinite(v)
                    for v in (x, y, width, height)
                )
                or width <= 0
                or height <= 0
            ):
                continue

            results.append({
                "annotation_id": ann.get("id"),
                "image_id": ann.get("image_id"),
                "category_id": ann.get("category_id"),
                "bbox": [x, y, width, height],
                "area_pixels_squared": width * height,
            })

        results.sort(key=lambda row: row["area_pixels_squared"])

        return json.dumps({
            "valid_boxes_considered": len(results),
            "smallest_boxes": results[:limit],
            "limitation": (
                "Small boxes are not necessarily incorrect. "
                "Visually inspect them before drawing conclusions."
            ),
        }, indent=2)

    # =====================================================
    # RAG DOCUMENT CREATION
    # =====================================================

    def _build_documents(self):
        documents = []

        images = self.coco_data.get("images", [])
        annotations = self.coco_data.get("annotations", [])
        categories = self.coco_data.get("categories", [])

        documents.append(
            "Dataset overview: " + json.dumps({
                "image_count": len(images),
                "annotation_count": len(annotations),
                "categories": categories,
            })
        )

        for ann in annotations:
            image = self.images_by_id.get(ann.get("image_id"))
            documents.append(
                "COCO annotation record: " + json.dumps({
                    "annotation": ann,
                    "image": image,
                })
            )

        for row in self.metrics_data:
            documents.append(
                "Training epoch metrics: " + json.dumps(row)
            )

        documents.append(
            "Bounding-box audit: " + self.audit_bounding_boxes()
        )
        documents.append(
            "Category distribution: "
            + self.audit_category_distribution()
        )
        documents.append(
            "Training metrics analysis: " + self.audit_training_logs()
        )

        return documents

    # =====================================================
    # RAG INITIALIZATION AND EMBEDDING CACHE (OFFLINE-SAFE)
    # =====================================================

    def _initialize_rag(self):
        self.documents = self._build_documents()

        if not self.documents:
            print("[!] No documents available for RAG.")
            return

        try:
            from sentence_transformers import SentenceTransformer

            print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
            try:
                self.embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                self.embedder = SentenceTransformer(EMBEDDING_MODEL)

            document_hash = hashlib.sha256(
                "\n".join(self.documents).encode("utf-8")
            ).hexdigest()

            if EMBEDDING_CACHE_PATH.exists():
                try:
                    with np.load(
                        EMBEDDING_CACHE_PATH, allow_pickle=False
                    ) as cached:
                        cached_hash = str(
                            cached["document_hash"].item()
                        )
                        cached_embeddings = cached["embeddings"].copy()

                    if (
                        cached_hash == document_hash
                        and len(cached_embeddings) == len(self.documents)
                    ):
                        self.document_embeddings = cached_embeddings
                        print(
                            "[OK] Loaded cached RAG embeddings: "
                            f"{len(self.documents)} documents."
                        )
                        return

                    print("[INFO] RAG cache outdated; rebuilding.")

                except Exception as exc:
                    print(f"[WARN] Could not load embedding cache: {exc}")

            print("[INFO] Creating RAG embeddings...")

            embeddings = self.embedder.encode(
                self.documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )

            self.document_embeddings = np.asarray(
                embeddings, dtype=np.float32
            )

            np.savez_compressed(
                EMBEDDING_CACHE_PATH,
                document_hash=np.array(document_hash),
                embeddings=self.document_embeddings,
            )

            print(
                "[OK] RAG embeddings saved to "
                f"{EMBEDDING_CACHE_PATH}"
            )

        except Exception as exc:
            self.embedder = None
            self.document_embeddings = None
            print("[WARN] RAG semantic embedding initialization skipped (Offline or connection error).")
            print("[INFO] Local shortcuts and tool calling remain fully functional.")

    # =====================================================
    # TOOL 5: SEMANTIC RETRIEVAL
    # =====================================================

    def retrieve_project_context(self, query):
        if self.embedder is None or self.document_embeddings is None:
            return json.dumps({
                "error": "RAG embeddings are unavailable (offline mode).",
                "suggestion": "Use direct tool shortcuts like 'audit bounding boxes' or 'analyze training loss'.",
            }, indent=2)

        try:
            query_embedding = self.embedder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

            scores = self.document_embeddings @ query_embedding

            indices = np.argsort(scores)[::-1][
                :min(MAX_RETRIEVAL_RESULTS, len(scores))
            ]

            results = []

            for index in indices:
                results.append({
                    "similarity": round(float(scores[index]), 4),
                    "document": self.documents[int(index)],
                })

            return json.dumps({
                "query": query,
                "retrieved_document_count": len(results),
                "retrieved_documents": results,
            }, indent=2)

        except Exception as exc:
            return json.dumps({
                "error": f"Retrieval failed: {exc}"
            }, indent=2)

    # =====================================================
    # GEMINI TOOL-CALLING INITIALIZATION
    # =====================================================

    def _initialize_chat(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            print(
                "[WARN] GEMINI_API_KEY is not set. "
                "Local audit shortcuts remain available."
            )
            return

        try:
            self.client = genai.Client(api_key=api_key)

            def audit_bounding_boxes_tool() -> str:
                """Validate COCO bounding boxes and image dimensions."""
                print("[TOOL CALLED] audit_bounding_boxes")
                return self.audit_bounding_boxes()

            def audit_category_distribution_tool() -> str:
                """Calculate annotation counts by category."""
                print("[TOOL CALLED] audit_category_distribution")
                return self.audit_category_distribution()

            def audit_training_logs_tool() -> str:
                """Analyze training and validation losses."""
                print("[TOOL CALLED] audit_training_logs")
                return self.audit_training_logs()

            def find_smallest_boxes_tool(limit: int = 10) -> str:
                """Find the smallest valid boxes by area."""
                print("[TOOL CALLED] find_smallest_boxes")
                return self.find_smallest_boxes(limit)

            def retrieve_project_context_tool(query: str) -> str:
                """Retrieve relevant local annotation and metrics records."""
                print("[TOOL CALLED] retrieve_project_context")
                return self.retrieve_project_context(query)

            self.tool_functions = [
                audit_bounding_boxes_tool,
                audit_category_distribution_tool,
                audit_training_logs_tool,
                find_smallest_boxes_tool,
                retrieve_project_context_tool,
            ]

            self.chat = self.client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    tools=self.tool_functions,
                    temperature=0.2,
                    max_output_tokens=1200,
                    system_instruction=(
                        "You are an AI/ML dataset auditor. "
                        "Use tools for exact numerical project facts. "
                        "For annotation records and dataset summaries, use "
                        "retrieve_project_context_tool. "
                        "For geometric validation, use "
                        "audit_bounding_boxes_tool. "
                        "For category counts, use "
                        "audit_category_distribution_tool. "
                        "For training metrics, use audit_training_logs_tool. "
                        "For the smallest boxes, use find_smallest_boxes_tool. "
                        "Never invent tool results. Clearly distinguish "
                        "geometric validity from visual correctness. "
                        "COCO boxes use [x, y, width, height] in pixels. "
                        "When resizing from original dimensions W,H to "
                        "new dimensions W2,H2, use sx=W2/W and sy=H2/H. "
                        "Transform x and width by sx; transform y and height "
                        "by sy. Answer completely but concisely."
                    ),
                ),
            )

            print(f"[OK] Gemini chat initialized: {GEMINI_MODEL}")

        except Exception as exc:
            self.client = None
            self.chat = None

            print("[WARN] Gemini initialization failed.")
            print("[ERROR TYPE]", type(exc).__name__)
            print("[ERROR DETAILS]", str(exc))

    # =====================================================
    # LOCAL COMMAND SHORTCUTS
    # =====================================================

    def _handle_local_shortcut(self, query):
        lower = query.lower().strip()

        if lower in {
            "audit bounding boxes",
            "check bounding boxes",
            "verify bounding boxes",
        }:
            print("[LOCAL AUDIT] Bounding-box validation")
            print(self.audit_bounding_boxes())
            return True

        if lower in {
            "analyze category imbalance",
            "analyse category imbalance",
            "audit category distribution",
            "check category distribution",
        }:
            print("[LOCAL AUDIT] Category distribution")
            print(self.audit_category_distribution())
            return True

        if lower in {
            "analyze training loss",
            "analyse training loss",
            "audit training logs",
            "check training metrics",
        }:
            print("[LOCAL AUDIT] Training metrics")
            print(self.audit_training_logs())
            return True

        if lower in {
            "find smallest boxes",
            "find the smallest boxes",
        }:
            print("[LOCAL AUDIT] Smallest boxes")
            print(self.find_smallest_boxes())
            return True

        return False

    # =====================================================
    # GEMINI REQUEST WITH RETRIES & QUOTA HANDLING
    # =====================================================

    def _ask_gemini(self, query):
        if self.chat is None:
            print(
                "[WARN] Gemini is unavailable. "
                "Use exact local audit commands instead."
            )
            return

        response = None

        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                print(
                    f"[INFO] Asking Gemini "
                    f"(attempt {attempt}/{MAX_API_ATTEMPTS})..."
                )

                response = self.chat.send_message(query)
                break

            except Exception as exc:
                error_text = str(exc)
                upper_error = error_text.upper()

                if "429" in upper_error or "RESOURCE_EXHAUSTED" in upper_error:
                    print("\n[QUOTA EXHAUSTED] Free tier daily request limit reached for Gemini.")
                    print("[TIP] You can instantly run your local tool shortcuts without any API limits:")
                    print("  -> Type: audit bounding boxes")
                    print("  -> Type: analyze category imbalance")
                    print("  -> Type: analyze training loss")
                    print("  -> Type: find smallest boxes\n")
                    return

                is_temporary = any(
                    marker in upper_error
                    for marker in (
                        "503",
                        "UNAVAILABLE",
                        "500 INTERNAL",
                        "502",
                        "504",
                        "DEADLINE_EXCEEDED",
                    )
                )

                if is_temporary and attempt < MAX_API_ATTEMPTS:
                    wait_seconds = 3 * attempt
                    print(
                        f"[WARN] Temporary API error. "
                        f"Retrying in {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                    continue

                print("\n[ERROR] Gemini request failed.")
                print("[ERROR TYPE]", type(exc).__name__)
                print("[ERROR DETAILS]", error_text)
                return

        if response is None:
            print("[WARN] Gemini did not return a response.")
            return

        answer = getattr(response, "text", None)

        if answer and answer.strip():
            print(f"\nAgent:\n{answer.strip()}\n")
        else:
            print(
                "[WARN] Gemini returned no text. "
                "Check the API response and tool-call logs."
            )

    # =====================================================
    # INTERACTIVE TERMINAL
    # =====================================================

    def run_interactive_auditor(self):
        print("\n--- Agentic RAG Data Auditor ---")
        print("Type 'exit' to quit.\n")

        print("Try these local audit shortcuts (bypasses API quotas instantly):")
        print("  audit bounding boxes")
        print("  analyze category imbalance")
        print("  analyze training loss")
        print("  find smallest boxes\n")

        while True:
            try:
                query = input("You: ").strip()

                if query.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                if self._handle_local_shortcut(query):
                    continue

                self._ask_gemini(query)

            except KeyboardInterrupt:
                print("\nExiting auditor.")
                break

            except Exception as exc:
                print("[ERROR] Unexpected error:", type(exc).__name__)
                print("[ERROR DETAILS]", str(exc))


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    auditor = AgenticRAGAuditor()
    auditor.run_interactive_auditor()