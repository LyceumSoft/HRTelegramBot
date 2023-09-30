import re
import random
with open("base.txt", "r", encoding="utf-8") as file:
    data = file.read()

data = re.sub(r"[^А-Яа-я\s]", "", data)
sentences = data.split("\n")

keyword_responses = {
    "привет": ["Привет!", "Здравствуйте!", "Добрый день!"],
    "компани": ["Smart Consulting — это самостоятельная компания, которая специализируется на создании IT-продуктов и консалтинге.", "Мы гордимся своими достижениями в мире технологий.", "Компания развивается с каждым днем, стремясь к новым горизонтам."],
    "ответственнос": ["Ответственность — одно из наших главных ценностей.", "Мы уделяем особое внимание выполнению своих обязательств перед клиентами и партнерами.", "Ответственность — это то, что делает нас успешными."],
    "смартократ": ["Смартократия - это основная интегральная политика Группы Компаний Smart Consulting.", "Мы строим наши отношения и процессы управления на основе ключевых принципов смартократии.", "Смартократия помогает нам достигать великих результатов и сохранять прозрачность в работе."],
    "роль": ["Роль - это ключевая структурная единица, определяющая функции и обязанности сотрудников в ГК.", "Каждая Роль имеет свой собственный набор обязательств и целей.", "Роли являются фундаментальным элементом организации ГК."],
    "политик": ["Политика - это набор установленных правил и процедур, которые регулируют деятельность сотрудников.", "Мы следуем политике управления Ролями, чтобы обеспечить эффективное функционирование ГК.", "Политика важна для поддержания структурированности и согласованности в работе."],
    "глоссар": ["Глоссарий содержит основные термины и определения, используемые в работе ГК.", "Мы регулярно обновляем глоссарий, чтобы удерживать терминологию актуальной.", "Глоссарий помогает устранить недоразумения и обеспечивает ясность в коммуникации."],
    "обязательст": ["Обязательства - это действия, которые каждый сотрудник берет на себя в рамках своей Роли.", "Обязательства помогают нам сфокусироваться на достижении целей и задач.", "Выполнение обязательств является ключевым аспектом нашей культуры работы."],
    "предназначен": ["Предназначение - это цель или назначение Роли, определяющее ее функции и задачи.", "Каждая Роль имеет уникальное предназначение, которое определяет ее вклад в общие цели ГК.", "Предназначение помогает нам понимать, какую роль выполняет каждый сотрудник."],
    "кодекс": ["Кодекс Смартократии - это основной документ, определяющий принципы управления и организации ГК.", "Мы строго следуем Кодексу в нашей работе и принятии решений.", "Кодекс обеспечивает стабильность и честность в управлении ГК."],
    "круг": ["Круг - это объединение нескольких Ролей для достижения общей цели или задачи.", "Иерархия кругов определяется их предназначением и значимостью для ГК.", "Круги способствуют организации работы и управлению проектами."],
    "лидеркру": ["Лидеркруга - это ключевая роль в управлении Кругом, ответственная за формирование и развитие команды.", "Лидеркруга играет важную роль в достижении Предназначения Круга.", "Лидеркруга утверждается Якорным кругом и активно взаимодействует с участниками Круга."],
    "фасилитаторкру": ["Фасилитатор круга - это роль, ответственная за модерирование и проведение встреч в Круге.", "Фасилитатор способствует более эффективным обсуждениям и принятию решений на встречах.", "Фасилитатор делает работу Круга более организованной и продуктивной."],
    "секретарькру": ["Секретарь круга - это роль, занимающаяся документированием и систематизацией информации о деятельности Круга.", "Секретарь играет важную роль в сохранении истории Круга и обеспечении доступности информации.", "Документация, созданная Секретарем, помогает участникам Круга оставаться информированными."],
    "домен": ["Домен - это область, которую контролирует Роль или Круг.", "Домен может быть назначен для управления определенными аспектами работы ГК.", "Решения о назначении Домена принимаются на Законодательной встрече Якорного круга с участием заинтересованных сторон."],
    "добровольное сообщество": ["Добровольное сообщество - это инициативное объединение сотрудников по личным или профессиональным интересам.", "Сообщества не регламентируются Кодексом и предоставляют сотрудникам возможность свободно обмениваться знаниями и опытом.", "Участие в добровольных сообществах способствует разнообразию и развитию ГК."],
    "ликвидация круг": ["Ликвидация Круга - это процедура, позволяющая завершить существование Круга, если это необходимо.", "Процедура ликвидации регламентируется Политикой и предусматривает определенные шаги и действия.", "Ликвидация Круга может быть инициирована сотрудниками, и она проводится в соответствии с установленными правилами."],
    "кандидатура на лид-линка": ["Кандидатура на Лид-линка - это предложение сотрудника на роль Лид-линка Круга.", "Участники Круга имеют право выдвигать кандидатов на эту роль.", "Лид-линка утверждают Акционеры ГК после рассмотрения кандидатур."],
    "полномочия лид-линка": ["Полномочия Лид-линка определяются Политикой и могут варьироваться в зависимости от Круга и его задач.", "Состав обязательств Лид-линка обсуждается и утверждается на Законодательной встрече Якорного круга.", "Лид-линк может выполнять свои обязанности параллельно с другими ролями."],
    "избираемая роль": ["Избираемая роль - это роль, на которую сотрудник избирается через голосование участников.", "Фасилитатор и Секретарь Круга - избираемые роли, их выборы проводятся в соответствии с Политикой проведения голосований и выборов.", "Избираемые роли играют важную роль в функционировании Кругов."],
    "зона ответственност": ["Зона ответственности - это область, за которую отвечает Роль.", "Домен закрепляется за Ролью или Кругом по решению Законодательной встречи.", "Зона ответственности помогает четко определить, за что отвечает каждая Роль и какие решения она принимает."]
}




def generate_response(question):
    question = question.lower()
    for keyword, responses in keyword_responses.items():
        if keyword in question:
            return random.choice(responses)

    return "Извините, не могу найти ответ на ваш вопрос."

def main():
    print("Добро пожаловать в HR-чатбот! (Введите 'выход' для завершения)")

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'выход':
            print("Чат завершен.")
            break

        response = generate_response(user_input)
        print(f"HR: {response}")

if __name__ == "__main__":
    main()
