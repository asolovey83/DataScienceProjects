import streamlit as st
import pandas as pd

st.title('Land Lot Prediction')
st.markdown('##### This is the final project for SkillFactory datascience course')


# Streamlit status
status = st.radio('What is your status', (3,4))
print(f'Your status is', status)

# Streamlit cadastre
cadastre = st.selectbox('Your cadastre number', [3222055100, 3222086800, 6525082500, 6525085600, 6523584000,
       6520687300, 6520687300, 6520686600, 6520680200, 5621455300,
       5621485000, 1423385000, 1222986000, 5321681000, 6520680200,
       6520680200, 6520680200, 6523584000, 6520683000, 6520683000,
       6520683000, 4825782700, 5924487600, 5924487600, 5924483800,
       5924483800, 5924483800, 5924483800, 5924483800, 5924483800,
       5924483800, 5924480400, 5924480400, 5925080800, 6525085600,
       6520686900, 6520687100, 6520687100, 6520687100, 6520687100,
       6520680200, 6520687300, 6520686900, 6520686900, 4820681200,
       4820981800, 4820981800, 4423382500, 4423382500, 5320483000,
       5322686200, 5325183200, 5924185400, 5924185400, 5924185400,
       5924185400, 5924185400, 5924189600, 5924189600, 5320686800,
       5924186100, 6824480500, 6523580700, 6520680200, 6520680200,
       6520684400, 6525085600, 6525082500, 6525085600, 6525085600,
       2621655700, 5322686200, 5325182600, 5325182600, 5325182600,
       6525085600, 6525085600, 6520985000, 6520985000, 6520685200,
       6520685200, 6525085600, 6525085600, 6520685200, 4620387000,
       5321681000, 5925080400, 5925080400, 5925080400, 5925080400,
       5925080400, 6824485000, 6824485000, 4820681200, 4820681200,
       4825182600, 4620381300, 7420385000, 7420385000, 5923883800,
       5924785600, 7422783500, 5324585100, 5320682800, 5324585100,
       5324585100, 5325182600, 5325185500, 5325182600, 5325182600,
       5325182600, 5924185400, 4825181200, 4825182600, 6520687100,
       5925081600, 5925081600, 5925080800, 5925080800, 4820981200,
       4820981800, 6523584000, 6525085600, 4423382500, 5321684900,
       5320481400, 5320483000, 5923883800, 5924787900, 5924787900,
       5924786300, 5922681700, 5922681700, 5922980400, 6324283500,
       6525085600, 6523584000, 6520687100, 6520686900, 6520686900,
       5922981200, 5922981200, 7422088400, 7422088400, 7422782500,
       7422782500, 5924480400, 5924480400, 7422782500, 7422782500,
       5924483800, 5924483800, 7420386000, 7420386000, 7420386000,
       5920380800, 5922985800, 5922985800, 5922985800, 5922985800,
       5925080400, 5925080400, 5925080400, 5925080400, 5925080400,
       4820980900, 4820982400, 4820982400, 4820981800, 4820985000,
       6525086000, 6525086000, 6523884500, 6523884500, 6523884500,
       6525085600, 6525086000, 6525085600, 6525085600, 6525086000,
       6525082500, 6525085600, 6525085600, 6525085600, 6525082500,
       6525082500, 6525085600, 6525085600, 6525085600, 6525085600,
       6523287700, 6523287700, 6523287700, 5322881100, 6523584000,
       6523580700, 6523584000, 4423382500, 1423385000, 5320485500,
       5925080800, 5925080800, 5925081600, 6520680200, 6520983700,
       6520687300, 5924185400, 5924185400, 5924182000, 5320483000,
       4820981200, 4820981800, 6525085600, 6525085600, 6525085600,
       6525085600, 6525085600, 6525085600, 6525085600, 6325786500,
       5924785600, 5922986200, 5922986200, 5922986200, 5922986200,
       5922985800, 5920388700, 4423382500, 4423382500, 5322686200,
       5621485500, 2322180400, 2322180400, 2322180400, 1222986000,
       1222986000, 5320483400, 5920384000, 5924487600, 5924487600,
       5925381200, 5925381200, 4820981800, 4820981800, 4820981800,
       5320482000, 5320483000, 5320483000, 5320483400, 5320481100,
       5320481100, 5320481100, 5924182300, 5924182300, 5924182000,
       5924182000, 5924182000, 5924182000, 5924182000, 5924182000,
       5924185400, 7422787500, 5325181700, 5325181700, 5322681500,
       6520687300, 6523584000, 6525085600, 6525085600, 6523584000,
       5920380400, 5920384000, 5922985800, 5925080400, 5925081600,
       5925081600, 5925080800, 5925080800, 5925082400, 5925082400,
       5925082400, 5925082400, 5925082400, 5920384000, 5322682800,
       5320484600, 5320486900, 5320481700, 6520686900, 6520685200,
       6520685200, 6520685200, 6520685200, 6520681200, 6520681800,
       6520681800, 6525085600, 1422783400, 1422783400, 1421586800,
       1421586800, 1824284700, 1824284700, 1824284700, 1824284700,
       1824284700, 1824284700, 5924182000, 5924182000, 5924182000,
       4820981800, 4820981800, 4820982400, 4820982400, 4820982400,
       4820982400, 6520683000, 6520686900, 6523584000, 6525085600,
       6525086000, 6525086000, 6525086000, 6525085600, 6525085600,
       6525085600, 6525086000, 6525085600, 6525085600, 6525483800,
       1222986000, 1222986000, 1222986000, 1222986000, 1222986000,
       1222986000, 1421586800, 5920380400, 3222083900, 3222083900,
       1824284000, 1824284000, 5320486900, 5320481700, 5321610100,
       5924486700, 7422782500, 5925080800, 5920386900, 5922387600,
       5922387600, 2624410100, 2624410100, 4423382500, 5924182000,
       5924182000, 5924182000, 5924187600, 5924187600])
print(f'Your cadastre number is', cadastre)


#Streamlit select test
koatuu = st.selectbox('Your Koatuu', ['Іванків, Іванківська, Вишгородський, Київська, Україна',
       'Іванківська, Вишгородський, Київська, Україна',
       'Подо-Калинівка, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Борозенська, Бериславський, Херсонська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Урожайне, Бериславська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Урожайне, Бериславська, Бериславський, Херсонська, Україна',
       'Раківка, Бериславська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Демидівка, Демидівська, Дубенський, Рівненська, Україна',
       'Демидівка, Демидівська, Дубенський, Рівненська, Україна',
       'Рудка, Демидівська, Дубенський, Рівненська, Україна',
       'Здовбицька, Рівненський, Рівненська, Україна',
       'Мар’їнська, Покровський, Донецька, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Верхня Ланна, Ланнівська, Полтавський, Полтавська, Україна',
       'Краснолуцька, Миргородський, Полтавська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Нововасилівка, Снігурівська, Баштанський, Миколаївська, Україна',
       'Нововасилівка, Снігурівська, Баштанський, Миколаївська, Україна',
       'Хибалівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Вересоч, Куликівська, Чернігівський, Чернігівська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Борозенська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Урожайне, Бериславська, Бериславський, Херсонська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Інгульська, Баштанський, Миколаївська, Україна',
       'Інгульська, Баштанський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Хибалівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Лохвицька, Миргородський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Андріяшівська, Роменський, Сумська, Україна',
       'Андріяшівська, Роменський, Сумська, Україна',
       'Градизька, Кременчуцький, Полтавська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Паплинці, Старосинявська, Хмельницький, Хмельницька, Україна',
       'Адампіль, Старосинявська, Хмельницький, Хмельницька, Україна',
       'Якушинецька, Вінницький, Вінницька, Україна',
       'Тавричанська, Каховський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Бериславська, Бериславський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Подо-Калинівка, Ювілейна, Херсонський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Чернелиця, Чернелицька, Коломийський, Івано-Франківська, Україна',
       'Чернелиця, Чернелицька, Коломийський, Івано-Франківська, Україна',
       'Старовірівська, Красноградський, Харківська, Україна',
       'Погребищенська , Вінницький, Вінницька, Україна',
       'Лохвицька, Миргородський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Калинівська, Бериславський, Херсонська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Великоолександрівська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Бродівська, Золочівський, Львівська, Україна',
       'Верхня Ланна, Ланнівська, Полтавський, Полтавська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Паплинці, Старосинявська, Хмельницький, Хмельницька, Україна',
       'Паплинці, Старосинявська, Хмельницький, Хмельницька, Україна',
       'Інгульська, Баштанський, Миколаївська, Україна',
       'Інгульська, Баштанський, Миколаївська, Україна',
       'Інгульська, Баштанський, Миколаївська, Україна',
       'Інгульська, Баштанський, Миколаївська, Україна',
       'Острівка, Куцурубська, Миколаївський, Миколаївська, Україна',
       'Бродівська, Золочівський, Львівська, Україна',
       'Бродівська, Золочівський, Львівська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Батуринська, Ніжинський, Чернігівська, Україна',
       'Батуринська, Ніжинський, Чернігівська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Краснолуцька, Миргородський, Полтавська, Україна',
       'Сергіївська, Миргородський, Полтавська, Україна',
       'Кардашівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Кардашівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Путивльська, Конотопський, Сумська, Україна',
       'Степанівська, Сумський, Сумська, Україна',
       'Степанівська, Сумський, Сумська, Україна',
       'Степанівська, Сумський, Сумська, Україна',
       'Вересоч, Куликівська, Чернігівський, Чернігівська, Україна',
       'Дрімайлівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Оболонська, Кременчуцький, Полтавська, Україна',
       'Глобинська, Кременчуцький, Полтавська, Україна',
       'Оболонська, Кременчуцький, Полтавська, Україна',
       'Оболонська, Кременчуцький, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Якушинецька, Вінницький, Вінницька, Україна',
       'Чорноморська, Миколаївський, Миколаївська, Україна',
       'Острівка, Куцурубська, Миколаївський, Миколаївська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Гребениківка, Боромлянська, Охтирський, Сумська, Україна',
       'Гребениківка, Боромлянська, Охтирський, Сумська, Україна',
       'Старовірівська, Красноградський, Харківська, Україна',
       'Червоне Озеро, Новослобідська, Конотопський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Богданівка, Прибузька, Вознесенський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Ланнівська, Полтавський, Полтавська, Україна',
       'Карлівка, Карлівська, Полтавський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Путивльська, Конотопський, Сумська, Україна',
       'Червоне Озеро, Новослобідська, Конотопський, Сумська, Україна',
       'Садівська, Сумський, Сумська, Україна',
       'Садівська, Сумський, Сумська, Україна',
       'Степанівська, Сумський, Сумська, Україна',
       'Кролевецька, Конотопський, Сумська, Україна',
       'Кролевецька, Конотопський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Старовірівська, Красноградський, Харківська, Україна',
       'Старовірівська, Красноградський, Харківська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Калинівська, Хмільницький, Вінницька, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лютенська, Миргородський, Полтавська, Україна',
       'Підлісне, Кіптівська, Чернігівський, Чернігівська, Україна',
       'Підлісне, Кіптівська, Чернігівський, Чернігівська, Україна',
       'Горбове, Куликівська, Чернігівський, Чернігівська, Україна',
       'Горбове, Куликівська, Чернігівський, Чернігівська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Новгород-Сіверська, Новгород-Сіверський, Чернігівська, Україна',
       'Новгород-Сіверська, Новгород-Сіверський, Чернігівська, Україна',
       'Горбове, Куликівська, Чернігівський, Чернігівська, Україна',
       'Горбове, Куликівська, Чернігівський, Чернігівська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Нововасилівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Батуринська, Ніжинський, Чернігівська, Україна',
       'Матіївка, Батуринська, Ніжинський, Чернігівська, Україна',
       'Матіївка, Батуринська, Ніжинський, Чернігівська, Україна',
       'Матіївка, Батуринська, Ніжинський, Чернігівська, Україна',
       'Бугрувате, Чернеччинська, Охтирський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Нечаянська, Миколаївський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Нижньосірогозька, Генічеський, Херсонська, Україна',
       'Нижньосірогозька, Генічеський, Херсонська, Україна',
       'Нижньосірогозька, Генічеський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Подо-Калинівка, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Подо-Калинівка, Ювілейна, Херсонський, Херсонська, Україна',
       'Подо-Калинівка, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Мирненська, Скадовський, Херсонська, Україна',
       'Мирненська, Скадовський, Херсонська, Україна',
       'Мирненська, Скадовський, Херсонська, Україна',
       'Лубенська, Лубенський, Полтавська, Україна',
       'Новооржицька, Лубенський, Полтавська, Україна',
       'Новооржицька, Лубенський, Полтавська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Тавричанська, Каховський, Херсонська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Мар’їнська, Покровський, Донецька, Україна',
       'Лютенська, Миргородський, Полтавська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Гребениківка, Боромлянська, Охтирський, Сумська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Борозенська, Бериславський, Херсонська, Україна',
       'Урожайне, Бериславська, Бериславський, Херсонська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Новгород-Сіверська, Новгород-Сіверський, Чернігівська, Україна',
       'Дрімайлівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Шевченківська, Куп’янський, Харківська, Україна',
       'Шевченківська, Куп’янський, Харківська, Україна',
       'Степанівська, Сумський, Сумська, Україна',
       'Степанівська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Червоне Озеро, Новослобідська, Конотопський, Сумська, Україна',
       'Путивльська, Конотопський, Сумська, Україна',
       'Червоне Озеро, Новослобідська, Конотопський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Хухра, Чернеччинська, Охтирський, Сумська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Лохвицька, Миргородський, Полтавська, Україна',
       'Хрінники, Демидівська, Дубенський, Рівненська, Україна',
       'Августинівка, Широківська, Запорізький, Запорізька, Україна',
       'Августинівка, Широківська, Запорізький, Запорізька, Україна',
       'Августинівка, Широківська, Запорізький, Запорізька, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Сергіївська, Миргородський, Полтавська, Україна',
       'Кардашівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Середино-Будська, Шосткинський, Сумська, Україна',
       'Шосткинська, Шосткинський, Сумська, Україна',
       'Шосткинська, Шосткинський, Сумська, Україна',
       'Шосткинська, Шосткинський, Сумська, Україна',
       'Дрімайлівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Дрімайлівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Дрімайлівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Дрімайлівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Сергіївська, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Хибалівка, Куликівська, Чернігівський, Чернігівська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Чорнухинська, Лубенський, Полтавська, Україна',
       'Лохвицька, Миргородський, Полтавська, Україна',
       'Лохвицька, Миргородський, Полтавська, Україна',
       'Урожайне, Бериславська, Бериславський, Херсонська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Тавричанська, Каховський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Урожайне, Бериславська, Бериславський, Херсонська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Бакирівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Кардашівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Бугрувате, Чернеччинська, Охтирський, Сумська, Україна',
       'Лебединська, Сумський, Сумська, Україна',
       'Тростянецька, Охтирський, Сумська, Україна',
       'Гребениківка, Боромлянська, Охтирський, Сумська, Україна',
       'Гребениківка, Боромлянська, Охтирський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Жигайлівка, Боромлянська, Охтирський, Сумська, Україна',
       'Жигайлівка, Боромлянська, Охтирський, Сумська, Україна',
       'Жигайлівка, Боромлянська, Охтирський, Сумська, Україна',
       'Жигайлівка, Боромлянська, Охтирський, Сумська, Україна',
       'Жигайлівка, Боромлянська, Охтирський, Сумська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Кардашівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Сенчанська, Миргородський, Полтавська, Україна',
       'Краснолуцька, Миргородський, Полтавська, Україна',
       'Гадяцька, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Бериславська, Бериславський, Херсонська, Україна',
       'Качкарівка, Милівська, Бериславський, Херсонська, Україна',
       'Качкарівка, Милівська, Бериславський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Новогродівська, Покровський, Донецька, Україна',
       'Новогродівська, Покровський, Донецька, Україна',
       'Хлібодарівська, Волноваський, Донецька, Україна',
       'Хлібодарівська, Волноваський, Донецька, Україна',
       'Батуринська, Ніжинський, Чернігівська, Україна',
       'Малинська, Коростенський, Житомирська, Україна',
       'Можари, Словечанська, Коростенський, Житомирська, Україна',
       'Можари, Словечанська, Коростенський, Житомирська, Україна',
       'Можари, Словечанська, Коростенський, Житомирська, Україна',
       'Можари, Словечанська, Коростенський, Житомирська, Україна',
       'Можари, Словечанська, Коростенський, Житомирська, Україна',
       'Можари, Словечанська, Коростенський, Житомирська, Україна',
       'Градизька, Кременчуцький, Полтавська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Калинівка, Березанська, Миколаївський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Березанська, Миколаївський, Миколаївська, Україна',
       'Тягинська, Бериславський, Херсонська, Україна',
       'Новорайська, Бериславський, Херсонська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Томарине, Бериславська, Бериславський, Херсонська, Україна',
       'Зеленопідська, Каховський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Щасливе, Ювілейна, Херсонський, Херсонська, Україна',
       'Скадовка, Чаплинська, Каховський, Херсонська, Україна',
       'Тавричанська, Каховський, Херсонська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Придніпровське, Червоногригорівська, Нікопольський, Дніпропетровська, Україна',
       'Хлібодарівська, Волноваський, Донецька, Україна',
       'Бакирівка, Чернеччинська, Охтирський, Сумська, Україна',
       'Чигиринська, Черкаський, Черкаська, Україна',
       'Іванківська, Вишгородський, Київська, Україна',
       'Іванківська, Вишгородський, Київська, Україна',
       'Малинська, Коростенський, Житомирська, Україна',
       'Листвин, Словечанська, Коростенський, Житомирська, Україна',
       'Листвин, Словечанська, Коростенський, Житомирська, Україна',
       'Листвин, Словечанська, Коростенський, Житомирська, Україна',
       'Гадяцька, Миргородський, Полтавська, Україна',
       'Великобудищанська, Миргородський, Полтавська, Україна',
       'Карлівка, Карлівська, Полтавський, Полтавська, Україна',
       'Стягайлівка, Зноб-Новгородська, Шосткинський, Сумська, Україна',
       'Горбове, Куликівська, Чернігівський, Чернігівська, Україна',
       'Боромлянська, Охтирський, Сумська, Україна',
       'Пологи, Чернеччинська, Охтирський, Сумська, Україна',
       'Червоне Озеро, Новослобідська, Конотопський, Сумська, Україна',
       'Чернеччина, Краснопільська, Сумський, Сумська, Україна',
       'Чернеччина, Краснопільська, Сумський, Сумська, Україна',
       'Успенка, Буринська, Конотопський, Сумська, Україна',
       'Рогатин, Рогатинська, Івано-Франківський, Івано-Франківська, Україна',
       'Рогатин, Рогатинська, Івано-Франківський, Івано-Франківська, Україна',
       'Новопсковська, Старобільський, Луганська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Роменська, Роменський, Сумська, Україна',
       'Старовірівська, Красноградський, Харківська, Україна'])

print(f'Your status is', koatuu)

#print('Hello')

