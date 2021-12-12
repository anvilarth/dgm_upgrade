# DGM upgrade



## Bishuk Anton changes
### Домашнее задание 1
1. Добавил пояснения для для теоретической задачи.
2. Поправил съехавшую формулу для второй теоретической задачи, также поменял формулы распределений, чтобы порядок был как в описании задания, поменял обозначения смеси для смеси распределения.
3. Поменял eps на vareps и чуть навел марафет.
4. Добавил комменты и тест к get_cross_entropy_loss.
5. Добавил комменты к MaskedLinear.
6. Добавил пояснения к тому, что происходит с нумерацией нейронов при кодировании.
7. Добавил сигнатуру функции get_mask в модель.
8. Добавил пояснения к тому, что нужно делать + некоторые комментарии.
9. Добавил рекомендацию к числу скрытых слоев.
10. Добавил наводящие вопросы к проверке работы сети + добавил явно что такое  двойка в выражении. Добавил рекомендации к параметрам обучения для обоих заданий.

### Домашнее задание 2
1. Поменял имптор tqdm, чтобы он в блокноте адекватно себя вёл.
2. Добавил комментарии в модель.
3. Добавил комментарии к гиперпараметрам.
4. Добавил комментарии в VAE и чуть изменил уже существующие.
5. Добавил рекомендации к параметрам обучения + чуть изменил комментарий.

### Домашнее задание 3
1. Поменял имптор tqdm, чтобы он в блокноте адекватно себя вёл.
2. Поменял описание get_normal.
3. Добавил комментов в энкодер, декодер и vae.
4. Исправил опечатку в AffineTransform.
5. Изменил prior в RealNVP.
6. Добавил рекомендации к параметрам.

### Домашнее задание 4
1. Перенес get_normal_kl.
2. Добавил рекомендации для параметров обучения.

### Домашнее задание 5
1. Добавил prior в генератор первой части.
2. Добавил рекомендации для параметров обучения первой части.
3. Изменил visualize_critic_output, чтобы прорисовывались точки данных.
4. Добавил замечание в gradient_penalty.
5. Добавил prior в генератор последней части.
6. Добавил рекомендации для параметров обучения второй и третьей части.



## Filatov Andrey
1. Добавил загрузку файлов
2. Вынес основные утилиты в файл utils
3. Minor fixes 

### TODO
- [ ] Сделать установку через pip install с нужными зависимостями
- [ ] Добавить логгирование в train_epoch/eval_model
- [ ] Стандартизировать форматирования prior, loss, sample
- [ ] Добавить USE CUDA как глобальный аргумент в нужные функции
