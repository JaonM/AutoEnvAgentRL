#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$project_dir/.venv/bin/python3" ]]; then
  export PATH="$project_dir/.venv/bin:$PATH"
fi

context=""
tag="env-factory-agent-sandbox"
start="false"
dockerfile=""
port="8080"
env_file=""
pin_only="false"

usage() {
  cat <<'EOF'
用法：scripts/build_docker_sandbox_image.sh --context DIR [选项]

构建 Docker 沙箱镜像。构建完成后默认不启动容器；构建前会验证 Dockerfile 第一条 FROM 镜像的 manifest；
原镜像不可达时，按顺序尝试平替镜像站点。

选项：
  --context DIR     Docker 构建上下文目录，必填
  --dockerfile FILE Dockerfile 路径，默认：DIR/Dockerfile
  --tag NAME        镜像名称，默认：env-factory-agent-sandbox
  --port N          宿主机端口，容器端口固定为 8000，默认：8080
  --env-file FILE   启动容器时注入的环境变量文件，默认：不使用
  --start           构建后启动容器，默认：不启动
  --pin-only        只将基础镜像固定到内容摘要，不构建镜像
  -h, --help        显示帮助

环境变量：
  SANDBOX_BASE_IMAGE          覆盖 Dockerfile 第一条 FROM 镜像；默认模板使用 docker.m.daocloud.io/library/python:3.14-slim
  SANDBOX_BASE_IMAGE_MIRRORS  空格分隔的平替镜像列表；默认自动尝试常用镜像站
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context|--dockerfile|--tag|--port|--env-file)
      if (( $# < 2 )); then echo "$1 需要提供参数值" >&2; exit 2; fi
      case "$1" in
        --context) context="$2" ;;
        --dockerfile) dockerfile="$2" ;;
        --tag) tag="$2" ;;
        --port) port="$2" ;;
        --env-file) env_file="$2" ;;
      esac
      shift 2
      ;;
    --start) start="true"; shift ;;
    --pin-only) pin_only="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$context" ]]; then
  echo "缺少 --context" >&2
  usage >&2
  exit 2
fi
if ! [[ "$port" =~ ^[1-9][0-9]*$ ]] || (( port > 65535 )); then
  echo "--port 必须是 1 到 65535：$port" >&2
  exit 2
fi
if [[ -z "$tag" ]]; then
  echo "--tag 不能为空" >&2
  exit 2
fi
if [[ -n "$env_file" && ! -f "$env_file" ]]; then
  echo "--env-file 文件不存在：$env_file" >&2
  exit 3
fi
if [[ "$context" != /* ]]; then context="$(pwd)/$context"; fi
if [[ -z "$dockerfile" ]]; then dockerfile="$context/Dockerfile"; fi
if [[ "$dockerfile" != /* ]]; then dockerfile="$(pwd)/$dockerfile"; fi
[[ -d "$context" ]] || { echo "构建上下文不存在：$context" >&2; exit 3; }
[[ -f "$dockerfile" ]] || { echo "Dockerfile 不存在：$dockerfile" >&2; exit 3; }
command -v docker >/dev/null 2>&1 || { echo "未找到 docker CLI" >&2; exit 4; }
docker_user="$(awk 'toupper($1) == "USER" {print $2; exit}' "$dockerfile")"
if [[ -z "$docker_user" || "$docker_user" == "root" || "$docker_user" == "0" ]]; then
  echo "Dockerfile 必须声明非 root USER：$dockerfile" >&2
  exit 4
fi
if grep -Eiq '^[[:space:]]*ENV[[:space:]].*(API_KEY|SECRET|TOKEN|PASSWORD)' "$dockerfile"; then
  echo "Dockerfile 不得写入 API key、secret、token 或 password" >&2
  exit 4
fi
from_image="$(awk 'toupper($1) == "FROM" {print $2; exit}' "$dockerfile")"
if [[ -z "$from_image" ]]; then
  echo "Dockerfile 中没有找到 FROM 指令：$dockerfile" >&2
  exit 5
fi
if [[ "$from_image" == *'${'* ]]; then
  from_image="python:3.14-slim"
fi
base_image="${SANDBOX_BASE_IMAGE:-$from_image}"

normalised="$base_image"
if [[ "$normalised" != */* || "$normalised" == docker.io/* ]]; then
  normalised="${normalised#docker.io/}"
  if [[ "$normalised" != */* ]]; then normalised="library/$normalised"; fi
fi
tag_part=""
if [[ "$normalised" == *@* ]]; then
  tag_part="@${normalised#*@}"
  normalised="${normalised%@*}"
elif [[ "$normalised" == *:* ]]; then
  tag_part=":${normalised##*:}"
  normalised="${normalised%:*}"
fi
default_mirrors=("mirror.gcr.io/${normalised}${tag_part}" "docker.m.daocloud.io/${normalised}${tag_part}")
if [[ -n "${SANDBOX_BASE_IMAGE_MIRRORS:-}" ]]; then
  read -r -a mirrors <<< "$SANDBOX_BASE_IMAGE_MIRRORS"
else
  mirrors=("${default_mirrors[@]}")
fi
candidates=("$base_image" "${mirrors[@]}")

selected=""
cached_base="false"
manifest_file="$(mktemp "${TMPDIR:-/tmp}/env-factory-manifest.XXXXXX")"
resolved_dockerfile="$(mktemp "${TMPDIR:-/tmp}/env-factory-dockerfile.XXXXXX")"
package_inventory="$(mktemp "${TMPDIR:-/tmp}/env-factory-packages.XXXXXX")"
smoke_cid_file="$(mktemp "${TMPDIR:-/tmp}/env-factory-smoke-cid.XXXXXX")"
rm -f "$smoke_cid_file"
cleanup() {
  if [[ -s "$smoke_cid_file" ]]; then
    smoke_cid="$(tr -d '[:space:]' < "$smoke_cid_file")"
    if [[ -n "$smoke_cid" ]]; then docker rm -f "$smoke_cid" >/dev/null 2>&1 || true; fi
  fi
  rm -f "$manifest_file" "$resolved_dockerfile" "$package_inventory" "$smoke_cid_file"
}
trap cleanup EXIT
read -r docker_os docker_arch < <(docker info --format '{{.OSType}} {{.Architecture}}')
if docker image inspect "$base_image" >"$manifest_file" 2>/dev/null; then
  selected="$(python3 "$script_dir/sandbox/resolve_container_image.py" \
    "$manifest_file" "$base_image" "$docker_os" "$docker_arch" --local-inspect || true)"
  if [[ -n "$selected" ]]; then
    cached_base="true"
    echo "使用平台匹配且带仓库摘要的本地缓存基础镜像：$selected" >&2
  fi
fi
if [[ -z "$selected" ]]; then
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    echo "验证基础镜像可达性：$candidate" >&2
    if docker manifest inspect --verbose "$candidate" >"$manifest_file" 2>/dev/null; then
      resolved="$(python3 "$script_dir/sandbox/resolve_container_image.py" \
        "$manifest_file" "$candidate" "$docker_os" "$docker_arch" || true)"
      if [[ -n "$resolved" ]]; then
        selected="$resolved"
        break
      fi
    fi
  done
fi
if [[ -z "$selected" ]]; then
  echo "原始基础镜像及平替镜像无法解析为当前平台的内容摘要：$base_image" >&2
  exit 6
fi
echo "使用基础镜像：$selected" >&2

awk -v image="$selected" '
  toupper($1) == "FROM" {
    line = $0
    sub(/^[[:space:]]*FROM[[:space:]]+[^[:space:]]+/, "FROM " image, line)
    print line
    next
  }
  { print }
' "$dockerfile" > "$resolved_dockerfile"

# Make the portable source identical to the context whose image is measured.
# In particular, COPY . . must not embed the earlier floating-tag Dockerfile.
cp "$resolved_dockerfile" "$dockerfile"
if [[ "$pin_only" == "true" ]]; then
  exit 0
fi
context_report="$(python3 "$script_dir/sandbox/validate_docker_context.py" "$context")" || {
  [[ -z "$context_report" ]] || echo "$context_report" >&2
  echo "Docker 构建上下文不符合可移植/敏感文件隔离契约：$context" >&2
  exit 3
}
echo "$context_report"
build_context_sha256="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["context_sha256"])' "$context_report")"
pull_flag="--pull"
if [[ "$cached_base" == "true" ]]; then pull_flag="--pull=false"; fi
docker build "$pull_flag" --file "$dockerfile" --tag "$tag" "$context"
# Keep a machine-readable image provenance record beside the sandbox.
image_id="$(docker image inspect --format '{{.Id}}' "$tag")"
image_user="$(docker image inspect --format '{{.Config.User}}' "$tag")"
image_os="$(docker image inspect --format '{{.Os}}' "$tag")"
image_arch="$(docker image inspect --format '{{.Architecture}}' "$tag")"
case "$docker_arch" in aarch64) docker_arch="arm64" ;; x86_64) docker_arch="amd64" ;; esac
case "$image_arch" in aarch64) image_arch="arm64" ;; x86_64) image_arch="amd64" ;; esac
if [[ "$image_os" != "$docker_os" || "$image_arch" != "$docker_arch" ]]; then
  echo "构建镜像平台不匹配：期望 $docker_os/$docker_arch，实际 $image_os/$image_arch" >&2
  exit 7
fi
if [[ -z "$image_user" || "$image_user" == "root" || "$image_user" == "0" ]]; then
  echo "构建后的镜像不是非 root 用户：$image_user" >&2
  exit 7
fi
echo "在生产安全边界内执行容器 pytest smoke test" >&2
docker run --rm --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --pids-limit 128 --memory 512m --cpus 1.0 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --tmpfs /app/.runtime:rw,nosuid,size=64m,uid=10001,gid=10001,mode=0700 \
  -e SANDBOX_TRAINER_API_KEY=container-smoke-test \
  -e SANDBOX_EVALUATOR_MOCK=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$tag" python3 -m pytest -q -p no:cacheprovider
echo "在相同安全边界内验证沙箱服务能够启动并响应健康检查" >&2
docker run -d --rm --cidfile "$smoke_cid_file" --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --pids-limit 128 --memory 512m --cpus 1.0 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --tmpfs /app/.runtime:rw,nosuid,size=64m,uid=10001,gid=10001,mode=0700 \
  -e SANDBOX_TRAINER_API_KEY=container-smoke-test \
  -e SANDBOX_EVALUATOR_MOCK=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$tag" >/dev/null
smoke_cid="$(tr -d '[:space:]' < "$smoke_cid_file")"
service_healthy=false
for _ in $(seq 1 50); do
  if docker exec "$smoke_cid" python3 -c \
    'import json,urllib.request; value=json.load(urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=1)); assert value.get("status") == "ok"'; then
    service_healthy=true
    break
  fi
  sleep 0.2
done
if [[ "$service_healthy" != "true" ]]; then
  docker logs "$smoke_cid" >&2 || true
  echo "沙箱服务未能在生产安全边界内启动" >&2
  exit 7
fi
docker rm -f "$smoke_cid" >/dev/null
: > "$smoke_cid_file"
echo "采集镜像内 Python 分发包 inventory" >&2
docker run --rm --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --pids-limit 128 --memory 512m --cpus 1.0 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  "$tag" python3 -c 'import importlib.metadata as m,json; values={}; [(values.__setitem__(str(d.metadata.get("Name") or "").strip(), d.version)) for d in m.distributions() if str(d.metadata.get("Name") or "").strip()]; print(json.dumps({"version":"1.0","packages":[{"name":name,"version":values[name]} for name in sorted(values,key=str.casefold)]},sort_keys=True,separators=(",",":")))' \
  > "$package_inventory"
python3 - "$package_inventory" "$context/python_packages.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(sys.argv[2]).write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
python3 - "$context/docker_image_metadata.json" "$tag" "$selected" "$image_id" \
  "$image_os" "$image_arch" "$image_user" "$dockerfile" "$context/requirements-dev.txt" \
  "$context/python_packages.json" "$build_context_sha256" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

Path(sys.argv[1]).write_text(json.dumps({
    "version": "5.0",
    "tag": sys.argv[2],
    "base_image": sys.argv[3],
    "image_id": sys.argv[4],
    "platform": {"os": sys.argv[5], "architecture": sys.argv[6]},
    "runtime_user": sys.argv[7],
    "dockerfile_sha256": digest(sys.argv[8]),
    "requirements_sha256": digest(sys.argv[9]),
    "python_packages_sha256": digest(sys.argv[10]),
    "build_context_sha256": sys.argv[11],
    "smoke_test": {
        "passed": True,
        "command": "python3 -m pytest -q -p no:cacheprovider",
        "network": "none",
        "read_only_root": True,
        "cap_drop": "ALL",
        "no_new_privileges": True,
        "non_root_user": True,
        "service_health": True,
        "runtime_tmpfs": {
            "path": "/app/.runtime",
            "uid": 10001,
            "gid": 10001,
            "mode": "0700",
        },
    },
    "built_at": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
if [[ "$start" == "true" ]]; then
  run_args=(
    --rm
    --read-only
    --cap-drop ALL
    --security-opt no-new-privileges:true
    --pids-limit "${SANDBOX_PIDS_LIMIT:-128}"
    --memory "${SANDBOX_MEMORY_LIMIT:-512m}"
    --cpus "${SANDBOX_CPU_LIMIT:-1.0}"
    --tmpfs /tmp:rw,noexec,nosuid,size=64m
    --tmpfs /app/.runtime:rw,nosuid,size=64m,uid=10001,gid=10001,mode=0700
  )
  for env_name in SANDBOX_TRAINER_API_KEY SANDBOX_LLM_API_KEY SANDBOX_LLM_BASE_URL SANDBOX_LLM_MODEL SANDBOX_LLM_TIMEOUT_SECONDS SANDBOX_LLM_MAX_RETRIES KIMI_K3_REASONING_EFFORT SANDBOX_EVALUATOR_MOCK; do
    if [[ -n "${!env_name:-}" ]]; then run_args+=(--env "$env_name"); fi
  done
  if [[ -n "$env_file" ]]; then run_args+=(--env-file "$env_file"); fi
  container_id="$(docker run -d "${run_args[@]}" --publish "$port:8000" \
    --mount "type=bind,source=$(cd "$context/data" && pwd),target=/workspace/data" \
    "$tag")"
  echo "沙箱容器已后台启动：$container_id"
fi
