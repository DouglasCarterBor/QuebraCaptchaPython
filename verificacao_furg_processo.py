from selenium import webdriver
from selenium.webdriver.common.by import By

navegador = webdriver.Chrome()
navegador.get("https://www.tjrs.jus.br/novo/busca/?tipoConsulta=por_processo&return=proc&client=wp_index&combo_comarca=&comarca=&numero_processo=&numero_processo_desktop=&CNJ=N&comarca=&nome_comarca=&uf_OAB=RS&OAB=&comarca=&nome_comarca=&nome_parte=")
navegador.find_element(By.XPATH,'document.querySelector("#num_processo_mask")').click()