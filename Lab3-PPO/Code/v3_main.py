from v3_ppo_agent_atari import v3_AtariPPOAgent

if __name__ == '__main__':

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		# "update_sample_count": 100, #test
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 256,
		"logdir": 'log/v3_1e8/',
		"update_ppo_epoch": 3,
		"learning_rate": 2e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.015,
		"horizon": 1024,
		"env_id": 'Enduro-v5',
		"render_fps": 30,
		# "eval_interval": int(1e5),
		"eval_interval": 1000,    # test
		"eval_episode": 3,
		"num_envs":64,

		"update_count": 10,
	    "final_entropy_coefficient": 0.0,
		"entropy_decay_steps": 1000000,

	}
	agent = v3_AtariPPOAgent(config)
	agent.train()


'''
2. 訓練步數 (training_steps)
當前設定: 1e9（10億步）
建議: 10億步是一個非常高的目標，可能需要長時間的訓練和大量的計算資源。通常，對於 Atari 遊戲，數千萬到數億步已足夠達到良好的性能。建議您根據硬體資源和訓練時間限制，將其設置為 1e7（1000萬步）或 1e8（1億步）作為起點，並根據訓練進展進行調整。
3. 更新樣本數 (update_sample_count)
當前設定: 10000
建議: 對於多環境並行（如64個環境），10000 代表每個環境約收集156步（10000 / 64 ≈ 156）。PPO 通常在收集數千步後進行一次更新，以獲得更穩定的梯度估計。建議將 update_sample_count 增加到 2048 或 4096，這樣每個環境約收集32到64步，從而提高梯度估計的穩定性。
4. 折扣因子 (discount_factor_gamma)
當前設定: 0.99
建議: 0.99 是 PPO 中的標準折扣因子，通常效果良好。不需要調整，除非有特定需求。
5. GAE λ (discount_factor_lambda)
當前設定: 0.95
建議: 0.95 是 GAE 的常見設置，平衡了偏差和方差。可以根據具體環境進行微調，但通常不需要大幅調整。
6. 剪裁係數 (clip_epsilon)
當前設定: 0.2
建議: 0.2 是 PPO 的標準值，通常效果良好。保持這個值是合理的。
7. 最大梯度範數 (max_gradient_norm)
當前設定: 0.5
建議: 0.5 是標準值，可以防止梯度爆炸。如果發現訓練不穩定，可以嘗試降低此值，例如 0.3 或 0.4。但一般來說，0.5 已經足夠。
8. 批量大小 (batch_size)
當前設定: 256
建議: 對於多環境並行（64環境），batch_size=256 是合理的。這意味著每次更新使用256個樣本。根據 update_sample_count 和 update_ppo_epoch 的設定，這個值應該是 update_sample_count 的因數。保持這個值，或者根據資源調整，但需保持 batch_size 為64的倍數。
9. 日誌目錄 (logdir)
當前設定: 'Lab3-PPO/Code/log/Enduro_v1_1e9/'
建議: 保持現有設定，確保日誌目錄存在且具有寫入權限。定期檢查日誌以監控訓練進展。
10. PPO 更新次數 (update_ppo_epoch)
當前設定: 3
建議: 3 是 PPO 的標準設定。可以根據具體情況調整，但通常不需要改變。
11. 學習率 (learning_rate)
當前設定: 2.5e-4
建議: 2.5e-4 是 PPO 的標準學習率。可以保持這個值，或者根據訓練穩定性進行微調。例如，根據訓練過程中損失的波動，可以考慮稍微降低到 1e-4 或增加到 3e-4。
12. 價值係數 (value_coefficient)
當前設定: 0.5
建議: 0.5 是常見的設置，平衡策略損失和價值損失。可以根據訓練結果調整，但通常不需要更改。
13. 熵係數 (entropy_coefficient)
當前設定: 0.01
建議: 0.01 是鼓勵策略探索的標準值。可以根據策略探索的需求調整，例如增加到 0.02 以增加探索，但需避免過高導致策略發散。
14. 時間步長 (horizon)
當前設定: 512
建議: 512 是較為標準的步數，用於收集樣本。根據 update_sample_count 和多環境數量，可以調整以確保策略更新時有足夠的樣本。例如，2048 或 4096 步可能更適合大環境數量。
15. 環境 ID (env_id)
當前設定: 'Enduro-v5'
建議: 確認這是正確的 Atari 環境 ID。確保環境 ID 和 Gym 版本匹配，避免前綴問題。正確的環境 ID 應為 ALE/Enduro-v5，如果已經在 make_env 函數中添加了 ALE/ 前綴，則 env_id 應為 Enduro-v5。
16. 渲染幀率 (render_fps)
當前設定: 30
建議: 這主要影響視覺化部分，不影響訓練。可根據需要設置，通常不需要調整以影響訓練性能。建議在訓練過程中禁用渲染以提高訓練速度，除非需要可視化。
17. 評估間隔 (eval_interval)
當前設定: 10^6（在 Python 中這實際上是 10 XOR 6 = 12）
問題: 10^6 在 Python 中表示位運算的 XOR 操作，結果為 12。如果您意圖設為 1e6（100萬步），應使用 1e6 或 1000000。
建議: 如果需要設置為1e6，請改為 1e6 或 1000000。這樣每100萬步進行一次評估，對於訓練過程來說是合理的。如果設置為12，可能評估太頻繁，影響訓練速度。
修正:
python
複製程式碼
"eval_interval": int(1e6),  # 或 "eval_interval": 1000000
18. 評估回合數 (eval_episode)
當前設定: 3
建議: 3 是適中的數量，用於估計策略性能。可以根據需要增加到 5 或更多，以獲得更穩定的評估結果。
19. 環境數量 (num_envs)
當前設定: 64
建議: 使用64個環境能夠有效地加速樣本收集，但需要足夠的計算資源（CPU核心、記憶體等）。如果發現資源瓶頸，可以減少環境數量，例如到32或16，觀察訓練速度和性能的變化。具體選擇應根據您的硬體資源進行調整。
'''
