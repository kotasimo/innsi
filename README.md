# ロードセル解析プロジェクト

大学のロードセル研究用のPython開発基盤です。実行はCodex Cloud、コード・設定・意思決定の正本はGitHub `kotasimo/-----` で管理します。研究室PC・個人ノートPCにはPythonのインストールは不要です。解析手法・数式は今後の指示で決定します。

## フォルダ構成

| パス | 用途 |
| --- | --- |
| `src/loadcell_analysis/` | 今後の解析プログラム |
| `tests/` | 手法決定後に追加するテスト |
| `scripts/` | 環境構築・動作確認 |
| `data/versioned/` | Git管理を許可した少量のCSV・サンプル |
| `data/external/` | Git管理しない原データ・大容量・機密データ |
| `results/` | Git管理しない生成結果 |
| `docs/decisions/` | 開発方針・決定事項・タスク引き継ぎ |
| `docs/references/` | Git管理を許可した資料と資料の所在情報 |

既存の `hammering/`、`共和/`、`日章/`、`ロードセル/`、`研究方針/` とルートの資料は従来資産として保持しています。既に追跡済みのCSV・画像・キャッシュは `.gitignore` だけでは管理対象から外れません。移動・削除・解析条件の変更は別途確認して実施します。

## クラウドでの準備

リポジトリのルートで実行します（現在は `/workspace/-----`）。Python 3.12を使用します。

```bash
bash scripts/setup.sh
.venv/bin/python scripts/check_environment.py
```

初回はPyPIから依存関係を取得します。主要依存は `requirements.txt`、推移的依存を含む固定版は `requirements-lock.txt` に記録しています。依存の更新時は両方を整合させ、動作確認して変更を記録します。仮想環境はタスク内専用でGitには保存しません。

環境確認ではNumPyのFFT、SciPyの信号処理、PandasのCSV読み込み、Matplotlibの画像出力を小さな合成データで確認します。研究の解析手法や実データの妥当性を検証するものではありません。常駐サービスは不要です。

## 2台のPC・別タスクへの引き継ぎ

1. 両PCで同じ権限のあるGitHub・Codexアカウントにログインします。CodexのGitHub接続でこのPrivateリポジトリへのアクセスを許可します。
2. Codex Cloudで同じリポジトリ・環境を選び、最新の `main` または継続中の作業ブランチからタスクを作成します。最初に `AGENTS.md` と `docs/decisions/` を読みます。
3. 作業は目的別ブランチで行い、変更・検証・未解決事項を記録します。Codexは終了時に作業対象のファイルを選んでcommitし、作業ブランチをGitHubへ自動でpushします。PRを通してレビュー・マージします。機密データの混入を差分で確認します。
4. もう一方のPCでは同じタスクを開いて継続するか、新しいタスクでpush済みブランチを選びます。別タスクの未commit・未pushファイルや `/workspace/shared` の内容は自動同期されません。
5. PRをレビューして `main` にマージします。既に動いている別タスクには自動反映されません。最新のブランチから新規タスクを開始するか、作業を保護してから明示的に取り込みます。

Git操作の例（ブランチ名・ファイルは作業に合わせて選択）：

```bash
git status --short
git switch -c work/project-foundation
git add README.md AGENTS.md requirements.txt requirements-lock.txt .gitignore scripts docs src tests data results
git diff --cached
git commit -m "Set up load cell development foundation"
git push -u origin work/project-foundation
```

履歴確認は `git log --oneline`、変更を取り消す場合は共有済み履歴を保持する `git revert <commit>` を使い、PRで反映します。`reset --hard` やforce pushを通常の復元手順にはしません。

## 解析手順の正本

具体的な解析手順は [研究方針/解析手順２.pdf](研究方針/解析手順２.pdf) を参照します。概要は、実測出力の波形確認とFFT、データから判断した任意モード数の伝達関数推定、伝達関数から微分方程式への変換、FFT係数で構成した出力フーリエ級数を使った入力の周期的な特解の推定です。最高使用周波数を4条件以上変えて比較し、各条件の入力波形・最大値・最小値・ピーク時刻を記録します。

インパルス入力の仮定、未校正入力をNと表記しないこと、解析区間・チャンネル・補正・周波数除外を明記することも必須です。解析式や実装の細部は、このPDFの内容と後続のユーザー指示を優先します。

## データと結果

`data/README.md` の基準で保存先を決めます。外部保存データはタスク開始時に安全な手段で `data/external/` に配置してください。GitHubや環境スナップショットは外部ストレージの代わりにはなりません。結果を他PCへ引き継ぐ場合は承認された保管先に保存し、条件・コードのcommit・入力データ識別子を記録します。共有可能な小さな最終成果だけを選んで `docs/` に保存できます。

## ユーザー側の初回操作

- GitHubリポジトリ画面で **Private** 表示を確認します。Publicの場合は Settings → General → Danger Zone → Change repository visibility でPrivateに変更します。
- CodexのGitHub接続で対象リポジトリへの権限を確認します。どちらのPCでも同じプロジェクトにアクセスできるアカウントを使います。
- この変更をGitHubへcommit・pushしPRをマージします。ファイルが未pushのままでは別タスクに引き継げません。
- Codexの環境設定で保存されたセットアップ・開始手順を確認して保存し、Publishします。公開された環境はGitHubリポジトリの公開設定とは別です。新規タスクで環境確認を実行し、復元を確認します。

GitHubの公開範囲の変更、環境のPublish、新規タスクへの復元は、この作業だけでは完了しません。
