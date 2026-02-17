# Artifact Bundle

Large generated artifacts are distributed separately from this git repository.

## Recommended bundle structure

```text
artifacts/
  logs/
  results/
  plots/
  checksums.txt
```

## Generate checksums

```bash
find artifacts -type f -print0 | sort -z | xargs -0 shasum -a 256 > artifacts/checksums.txt
```

## Verify checksums

```bash
shasum -a 256 -c artifacts/checksums.txt
```
