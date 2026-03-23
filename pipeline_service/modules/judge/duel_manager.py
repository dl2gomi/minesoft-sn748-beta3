import time
from typing import Optional, Tuple

from logger_config import logger
from modules.grid_renderer.render import GridViewRenderer
from schemas.types import ImageTensor
from .judge_pipeline import JudgePipeline
from .schemas import JudgeInput, JudgeOutput


class DuelManager:
    """Orchestrates mesh duels using a provided judge pipeline."""

    def __init__(self, renderer: Optional[GridViewRenderer] = None) -> None:
        self.renderer = renderer

    async def run_duel(
        self,
        pipeline: JudgePipeline,
        prompt_image: ImageTensor,
        img1_grid: ImageTensor,
        img2_grid: ImageTensor,
        seed: int,
    ) -> Tuple[int, str]:
        """
        Run a position-balanced duel between two candidate grid images.

        Args:
            pipeline: Loaded judge pipeline used for inference.
            prompt_image: Original prompt image as ImageTensor.
            img1_grid: First candidate grid view as ImageTensor.
            img2_grid: Second candidate grid view as ImageTensor.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (winner_idx, issues):
                winner_idx: -1 if img1 wins, 1 if img2 wins.
                issues: Human-readable issue summary.
        """
        logger.debug("Running position-balanced VLLM duel...")
        res_direct  = await pipeline.judge(prompt_image, img1_grid, img2_grid, seed)
        res_swapped = await pipeline.judge(prompt_image, img2_grid, img1_grid, seed)

        score1 = (res_direct.penalty_1 + res_swapped.penalty_2) / 2
        score2 = (res_swapped.penalty_1 + res_direct.penalty_2) / 2
        issues = (
            f"| Direct: {res_direct.issues} | Swapped: {res_swapped.issues}"
            if res_direct.issues or res_swapped.issues
            else ""
        )

        # Lower penalty = better; draw defaults to second candidate
        winner = -1 if score1 < score2 else 1

        logger.debug(
            f"Duel scores — Candidate 1: {score1:.1f} (direct={res_direct.penalty_1}, swapped={res_swapped.penalty_2}) | "
            f"Candidate 2: {score2:.1f} (direct={res_direct.penalty_2}, swapped={res_swapped.penalty_1}) | Winner: {winner}"
        )

        return winner, issues

    async def judge_grid_views(self, request: JudgeInput) -> JudgeOutput:
        """
        Judge a list of grid views by comparing with the judge pipeline.

        Args:
            request: JudgeInput with pipeline, grid views, prompt image, and seed.

        Returns:
            JudgeOutput with winner_index.
        """
        t1 = time.time()

        if len(request.grid_views) < 2:
            logger.warning("Less than 2 grid views provided to judge")
            return JudgeOutput(winner_index=0)

        logger.info(f"Judging {len(request.grid_views)} grid views with prompt image")

        best_idx = 0
        for i in range(1, len(request.grid_views)):
            winner, issues = await self.run_duel(
                request.pipeline, request.prompt_image, request.grid_views[best_idx], request.grid_views[i], request.seed
            )
            logger.info(f"Duel [{best_idx} vs {i}] → winner: {i if winner == 1 else best_idx} {issues}")
            if winner == 1:  # Image 2 wins → update best candidate
                best_idx = i

        logger.success(
            f"Judging {len(request.grid_views)} grid views with prompt image took {time.time() - t1:.2f}s | Winner: {best_idx}"
        )

        return JudgeOutput(winner_index=best_idx, grid_view_winner=request.grid_views[best_idx])
