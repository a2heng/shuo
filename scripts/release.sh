#!/usr/bin/env bash
# 说 · Shuo — 发版脚本（只打 tag，不写版本号）
#
# 版本号规则：日期 YY.MM.DD（取本机日期），tag = v<版本>。
# 同一天第二次发版自动用 -2、-3…（semver 预发布段；deb 里等同修订号），避免撞 tag。
#
# 版本号**只存在于 tag**：CI 读到 tag 后导出 VERSION，再用
# `scripts/set-version.sh` 写进构建配置，并用于产物命名 / Release 标题。
# 所以这里不修改、不提交任何版本字段。
#
# 用法：./scripts/release.sh
#   1) 检查在 main、工作区干净、远端 tag 已同步
#   2) 按当天日期算出 tag（同日重复自动加 -2）
#   3) 打 tag 并推送
#   4) GitHub Actions（.github/workflows/release.yml）自动构建
#      Windows NSIS + Linux deb/AppImage，并创建同名 Release
#
# 只构建不发版：GitHub Actions 页面手动 workflow_dispatch
# （Release 作业只在 tag 触发时运行）。
set -euo pipefail

cd "$(dirname "$0")/.."

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "请在 main 上发版（当前分支：$branch）" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "工作区不干净，先提交或还原：" >&2
  git status --short >&2
  exit 1
fi

# 远端 tag 可能是别的机器推的，先同步再算版本
git fetch --tags --quiet origin

# 版本号必须能被 semver 解析：个位月/日不能带前导零（26.11.03 → 26.11.3）
base=$(date +%y.%-m.%-d)
version=$base
i=1
while git rev-parse -q --verify "refs/tags/v$version" >/dev/null; do
  i=$((i + 1))
  version="$base-$i"
done

git tag -a "v$version" -m "说 · Shuo $version"
git push origin main
git push origin "v$version"

url=$(git remote get-url origin |
  sed -E 's#^git@github\.com:#https://github.com/#; s#^https://github\.com/#https://github.com/#; s#\.git$##')

echo
echo "已推送 v$version（版本号由 CI 从 tag 注入，仓库里不写死）"
echo "  发版：$url/releases/tag/v$version"
echo "  构建：$url/actions/workflows/release.yml"
