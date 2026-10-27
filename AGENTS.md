# Codex 共通指示

- このプロジェクトは大学研究用ロードセル解析。ユーザーとのやり取りと研究記録は原則日本語。
- まず README.md、docs/decisions/、git status を確認する。既存変更を保護する。
- 各Codex Cloudタスクは独立している。既存checkoutを使い、ユーザーの明示要求なしにGit worktreeを作成しない。
- GitHubがコード・設定・決定事項の正本。別タスクの変更は自動同期されない。ブランチ・commit・PRと未解決事項を引き継ぐ。
- Python 3.12。環境構築は bash scripts/setup.sh、確認は .venv/bin/python scripts/check_environment.py。ローカルPCの環境や /workspace/shared のみに依存しない。
- 解析方法、数式、フィルタ条件、校正値、単位、サンプリング解釈を独自に決めない。今後のユーザー指示を待つ。環境確認の合成データ処理を研究解析と混同しない。
- 新しいコードは src/loadcell_analysis/ に置く。解析手法が指定されたら意味のあるテストを tests/ に追加する。
- 既存の研究ファイル・スクリプト・解析結果を無断で移動・削除・上書きしない。従来スクリプトは出力先が固定の場合があるため実行前に確認する。
- 原データは不変に扱う。Git管理は data/versioned/、外部管理は data/external/、生成結果は results/。秘密情報・認証情報・機密測定データをcommitしない。
- .gitignoreは既に追跡済みのファイルには効かない。追跡解除や履歴の書き換えは別途合意して行う。
- 設定・依存更新は再現性を保ち requirements.txt と requirements-lock.txt の整合を確認。TLS・署名・チェックサム検証を無効にしない。
- 方針変更は docs/decisions/ に背景・決定・影響を記録。終了時は変更、実行した検証、未実施項目、次の作業を報告する。
- 共有済み履歴の復元は git revert を基本とし、破壊的なresetやforce pushを独断で行わない。

- ユーザーの継続指示：コード・設定・文書の変更作業は、関連する検証と差分確認後、作業分を選択してcommitし、作業ブランチをGitHubへpushするところまで自動で行う。通常のcommit・pushに毎回確認を求めない。既存の無関係な変更や秘密情報を含めない。mainへのマージ、force push、履歴書き換えはこの許可に含まれない。pushが拒否された場合は原因を報告し、未pushを完了と扱わない。
