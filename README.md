- Python **>= 3.13**
- (опционально) [Graphviz](https://graphviz.org/) (`dot`) — для экспорта DFA в `.dot` / `.svg` при запуске `screening_analyzer`

## Установка на GNU/Linux

```bash
cd vibescreening
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Примеры

```bash
python -m refal.sema.analyze refal/test/refal_screenings/simple.ref
```

Анализ строки без файла:

```bash
python -m refal.sema.analyze --string 'f { s.1 = ; s.1 = ; }'
```
