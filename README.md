В dist находится SAR Jammer.exe - файл для запуска программы.

В source_code содержатя исходные файлы (src/) и список зависимостей (requirements.txt).

Для устанвоки зависимостей откройте терминал и укажите путь до папки с файлами, далее создайте виртуальное окружение:

python -m venv venv # Windows

python3 -m venv venv # macOs/Linux

Активируйте виртуальное окружение:

venv/Scripts/activate.bat # Windows

source venv/bin/activate

Установите зависимости из файла requirements.txt:

pip install -r requirements.txt