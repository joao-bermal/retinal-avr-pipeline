# Plano de implementação — treinos reais em GPU + API (opcional)

Escrito em 25/09/2026, execução autônoma autorizada pelo usuário (ele deixou o PC
rodando e saiu). Contexto: nesta sessão consertamos a ROI peripapilar (Zona B/disco
óptico) do pipeline AVR, consolidamos o código em `retinal-avr-pipeline`, e depois
resolvemos o ROCm (Secure Boot desabilitado, driver amdgpu do kernel 6.17 + ROCm 6.4.2
userspace instalado). Agora a GPU (RX 6800XT) está de fato disponível via
`.venv` (`torch==2.9.1+rocm6.4`). Este documento é o plano de execução dos próximos
passos, para eu seguir sem precisar de aprovação passo a passo.

## Nota de performance (importante)

MIOpen faz busca exaustiva de kernel na primeira vez que vê uma combinação nova de
forma/tamanho (pode levar 1h+ por camada única). Todas as invocações de treino/inferência
a partir de agora usam:
```bash
export MIOPEN_FIND_MODE=FAST
export MIOPEN_FIND_ENFORCE=SEARCH
```
Isso troca alguns % de performance de kernel por não travar em autotuning exaustivo.

## Ordem de execução

### 1. Treino real do detector de disco óptico (prioridade máxima)
Esse é o único modelo que só teve smoke-test em CPU (2-3 épocas, dice ~0.06-0.86
instável, dataset pequeno). Comando:
```bash
python main.py --train_od
```
- Config: `OPTIC_DISC_CONFIG` (80 épocas, early stopping patience=15, batch=2).
- Meta: dice >= 0.85 (disco é uma forma compacta, meta razoável).
- Ao concluir: rodar a validação qualitativa (centro/raio detectado vs `mask_OD` real)
  que já fiz manualmente antes, comparando contra a heurística de CV pura, pra
  confirmar que o modelo treinado é realmente melhor que o fallback.

### 2. Treino real da segmentação vascular (Enhanced U-Net)
O checkpoint atual (`dice=0.7971`) já bate a meta do TCC (0.7965), mas foi treinado em
condições variadas (CPU/GPU mistos, ambientes diferentes). Vale gerar um checkpoint
limpo e rastreável a partir do codebase consolidado:
```bash
python main.py --train_seg
```
- Config: `SEGMENTATION_CONFIG` (150 épocas, early stopping patience=25, batch=4).
- Se bater ou superar 0.7965 de Dice, vira o novo checkpoint "oficial" (o
  `_find_latest_checkpoint` do pipeline já pega o mais recente por mtime automaticamente).

### 3. Treino real da classificação A/V (Multi-Dataset AV-Net)
Checkpoint atual já tem macro F1 0.9538 (bem acima da meta 0.78). Ainda assim, треino
completo (250 épocas, early stopping patience=30) pra ter um checkpoint gerado
integralmente por este codebase/GPU, sem o histórico misto:
```bash
python main.py --train_av
```

### 4. Validação end-to-end pós-treino
- `python tests/sanity_check.py` e `python tests/test_avr_calculator.py`.
- `python main.py --run_pipeline data/DRIVE/test/images/01_test.tif` — confirmar que
  `optic_disc_method` passa a reportar `TRAINED_MODEL` (não mais `CV_BRIGHTEST_REGION`
  nem `FALLBACK_IMAGE_CENTER`) com confiança razoável.
- Reexecutar o script de validação do disco óptico contra `data/IOSTAR/mask_OD/*`
  (distância centro-a-centro), comparando modelo treinado vs. heurística de CV (que
  tinha mediana ~224px de erro — o modelo treinado deve ficar bem abaixo disso).
- Commit local dos novos checkpoints? **Não** — `models/` está no `.gitignore`
  (pastas grandes). Só documentar os números finais no commit da API/relatório final.

### 5. (Se der tempo) API web para rodar o pipeline

Objetivo: expor `ScientificAVRPipeline.process_image()` via HTTP, pra poder rodar de
uma aplicação web (upload de imagem → JSON com AVR/CRAE/CRVE/risco/disco óptico).

Escopo mínimo (FastAPI, sem frontend):
- Novo `api/main.py` (ou `src/api/app.py`): endpoint `POST /analyze` recebendo upload
  de imagem, rodando `pipeline.process_image()`, devolvendo o dict de resultados em
  JSON (convertendo numpy/tensor pra tipos serializáveis).
- Endpoint `GET /health` simples.
- Carregar o pipeline uma vez no startup (não por request).
- `requirements-api.txt` ou adicionar `fastapi`+`uvicorn`+`python-multipart` ao
  `requirements-rocm.txt`.
- Sem autenticação/deploy real — é só a camada HTTP pra permitir integração futura
  com uma aplicação web, rodando localmente (`uvicorn api.main:app --reload`).
- Testar com `curl -F "image=@data/DRIVE/test/images/01_test.tif" http://localhost:8000/analyze`.

## Fora de escopo (mesmo com autonomia)

- Push pro GitHub — seguindo a mesma regra desta sessão inteira, isso fica pra quando
  o usuário confirmar explicitamente, mesmo com a autorização de seguir sem permissão
  pros próximos passos (treino/API), porque é uma ação que expõe código publicamente.
- Deploy da API em produção/nuvem.
- Mexer em configurações de sistema/BIOS de novo.

## Registro de progresso

(Vou atualizar esta seção conforme cada etapa terminar, com números reais.)

- [ ] Disco óptico treinado
- [ ] Segmentação retreinada
- [ ] Classificação A/V retreinada
- [ ] Validação end-to-end pós-treino
- [ ] API (se der tempo)
